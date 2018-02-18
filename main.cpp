#include <QCoreApplication>
#include <iostream>
#include <opencv/cv.hpp>
#include <regression.h>
#include <pcl/common/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/cloud_iterator.h>

/* Programm to replicate OPENCV part, without shifting algorithmus */
using namespace std;
#define NUM_FRAMES 128
#define SCALE_X 2.7
#define SCALE_Y 2.4
#define SCALE_Z 3.0
void generatePoint(pcl::PointXYZ& point, float x, float y, float z, float width, float height) {
    point.x = (float)x / width * SCALE_X;
    point.y = (float)y / height * SCALE_Y;
    point.z = (float)z / NUM_FRAMES * SCALE_Z;
}



void MatToPointXYZ(cv::Mat& openCVPointCloud, cv::Mat& labelInfo, std::vector<cv::Point>& eclipsePoints, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width) {

    int x = labelInfo.at<int>(0, cv::CC_STAT_LEFT);
    int y = labelInfo.at<int>(0, cv::CC_STAT_TOP);
    int labelWidth = labelInfo.at<int>(0, cv::CC_STAT_WIDTH);
    int labelHeight = labelInfo.at<int>(0,cv::CC_STAT_HEIGHT);
    int leftHeight = 0;
    int rightHeight = 0;
    for (int i = x; i < x + labelWidth ; i++ ){
        bool firstNotFound = true;
        int lastPointPosition = 0;
        for (int j = y; j < y + labelHeight; j++) {
            if (openCVPointCloud.at<unsigned char>(j,i) >=1.0f) {
                if (firstNotFound) {
                    firstNotFound = false;
                }
                lastPointPosition =j;
                if (i == x) {
                    leftHeight = j;

                }
                if (i == x + labelWidth -1) {
                    rightHeight = j;
                }
            }
            if(!firstNotFound) {
                pcl::PointXYZ point;
                generatePoint(point, i , lastPointPosition,z, width, height);
                point_cloud_ptr->points.push_back(point);
                eclipsePoints.push_back(cv::Point(i,lastPointPosition));
            }
        }
    }
}


void processOTCFrame(cv::Mat imageGray,int number , boost::shared_ptr<std::vector<std::tuple<int,int, cv::Mat, cv::Mat>>> needle_width  ) {

    cv::Mat transposedOCTimage;
    cv::flip(imageGray, imageGray, 0);
    cv::Mat thresholdedImage;
    cv::threshold(imageGray,thresholdedImage,0.26*255,1,0);

    cv::Mat filteredImage;
    cv::medianBlur(thresholdedImage, filteredImage, 3);

    cv::Mat labeledImage;
    cv::Mat labelStats;
    cv::Mat labelCentroids;

    int numLabels = cv::connectedComponentsWithStats(filteredImage,labeledImage, labelStats, labelCentroids);
    for (int i = 1 ; i< numLabels; i++) {
        if (labelStats.at<int>(i,cv::CC_STAT_AREA)>250) {
            cv::Mat labelInfo = labelStats.row(i);
            needle_width->push_back(std::tuple<int, int, cv::Mat, cv::Mat>(number, labelStats.at<int>(i, cv::CC_STAT_WIDTH), filteredImage, labelInfo));
        }
    }


}




boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> recognizeOTC(pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points, std::string oct_dir, bool only_tip ) {
    int fileCount = 128;
    int minFrameNumber = 0;
    int maxFrameNumber = fileCount;

    boost::shared_ptr<std::vector<std::tuple<int,int, cv::Mat, cv::Mat>>> needle_width(new std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>);
    cv::Mat imageGray;
    {
        pcl::ScopeTime t("Process OCT images");
        for (int number = minFrameNumber ; number < maxFrameNumber ; number ++) {
            std::stringstream filename;
            if (number <100) {
                filename <<"0";
            }
            if (number <10) {
                filename <<"0";
            }
            filename <<number<<".bmp";
            imageGray = cv::imread("/home/tuan/Desktop/Back up/BA/042801/"+filename.str(),CV_LOAD_IMAGE_GRAYSCALE);
            processOTCFrame(imageGray,number,needle_width);
            cv::waitKey(10);

        }
        int endIndex = needle_width->size();
        if (only_tip) {
            endIndex= regression(needle_width);
        }
        for (int w = 0 ; w < endIndex; w++) {
            std::tuple<int,int,cv::Mat, cv::Mat> tup = needle_width->at(w);
            std::vector<cv::Point> elipsePoints;
            MatToPointXYZ(std::get<2>(tup), std::get<3>(tup), elipsePoints, std::get<0>(tup), point_cloud_ptr, imageGray.rows, imageGray.cols);
            if (elipsePoints.size()>50) {
                cv::RotatedRect elipse = cv::fitEllipse(cv::Mat(elipsePoints));
                pcl::PointXYZ peak;
                generatePoint(peak,elipse.center.x, elipse.center.y, std::get<0>(tup), imageGray.cols, imageGray.rows );
                peak_points->push_back(peak);

            }
        }
    }
    float VOXEL_SIZE_ICP_  = 0.02f;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
    voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
    voxel_grid_icp.filter(*point_cloud_ptr);
    return needle_width;
}

void nicePrintInfo(cv::Mat file ) {
    cout<< "Value at CC_STAT_LEFT: " << file.at<int>(0,cv::CC_STAT_LEFT) << std::endl;
    cout<< "Value at CC_STAT_TOP: " << file.at<int>(0,cv::CC_STAT_TOP)<< std::endl;
    cout<< "Value at CC_STAT_WIDTH: " << file.at<int>(0,cv::CC_STAT_WIDTH)<< std::endl;
    cout<< "Value at CC_STAT_HEIGHT: " << file.at<int>(0,cv::CC_STAT_HEIGHT)<< std::endl;
}

void showModifiedPictures() {
    cv::Mat original = cv::imread("/home/tuan/Downloads/Test.jpg",CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat file1 = cv::imread("/home/tuan/Downloads/Test.jpg",CV_LOAD_IMAGE_UNCHANGED);
    for (int r = 0 ; r < file1.rows; r++) {
        for (int c =0 ; c <file1.cols; c++) {
            file1.at<cv::Vec3b>(r,c)[0]+=25;
            file1.at<cv::Vec3b>(r,c)[1]+=25;
            file1.at<cv::Vec3b>(r,c)[2]+=25;
        }
    }
    nicePrintInfo(file1);
    int x = file1.at<int>(0, cv::CC_STAT_LEFT); //Tuan : left most point
    int y = file1.at<int>(0, cv::CC_STAT_TOP);


    cv::imshow("Orignal",original);
    cv::imshow("Woaw",file1);
    cv::waitKey();
    cv::namedWindow("Color",CV_WINDOW_AUTOSIZE);
    cv::moveWindow("Color",1000, 1000);
}


std::vector<float> convertVector3fToCl(Eigen::Vector3f vector3f) {
  std::vector<float> result;
  result.push_back(vector3f[0]);
  result.push_back(vector3f[1]);
  result.push_back(vector3f[2]);
}

std::vector<float> convertVector4fToCl(Eigen::Vector4f vector4f) {
    std::vector<float> result;
    result.push_back(vector4f[0]);
    result.push_back(vector4f[1]);
    result.push_back(vector4f[2]);
    result.push_back(vector4f[3]);
    return result;
}
/*
std::vector<std::vector<float>> convertMatrix4fToCL(Eigen::Matrix4f matrix4f) {
  std::vector<std::vector<float>> matrixVectors;
  std::vector<float> column1(matrix4f.cols(0));
  matrixVectors.push_back(column1);
  std::vector<float> column2 = matrix4f.cols(1);
  matrixVectors.push_back(column2);
  std::vector<float> column3 = matrix4f.cols(2);
  matrixVectors.push_back(column3);
  std::vector<float> column4 = matrix4f.cols(3);
  matrixVectors.push_back(column4);
  char3 sa;
  return matrixVectors;
}
*/
//https://stackoverflow.com/questions/12426061/how-to-pass-and-access-c-vectors-to-opencl-kernel
std::vector<float> convertPointXYZtoCL(pcl::PointXYZ point) {
    std::vector<float> result(point.x, point.y,point.z);
    return result;
}

std::vector<float> convertMatrix3fToCL(Eigen::Matrix3f matrix3f) {
  std::vector<float> vector(matrix3f.data(),matrix3f.data()+ matrix3f.cols()*matrix3f.rows());
  return vector;
}

std::vector<std::vector<float>> convertPointCloudToCL(pcl::PointCloud<pcl::PointXYZ> pointCloud) {
  std::vector<std::vector<float>> pointCloud;
  int size = pointCloud.size();
  for (int i = 0 ; i <size ; i++) {
    pointCloud.push_back(convertPointXYZtoCL(pointCloud.get(i)));
  }
}

//

int main()
{
    //http://pointclouds.org/documentation/tutorials/cloud_viewer.php
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>); //Create pointer to a points cloud of PointXYZ Type (3 Axis x,y,z)
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_not_cut(new pcl::PointCloud<pcl::PointXYZ>);  //Create pointer to a points cloud of PointXYZ Type (3 Axis x,y,z)
    pcl::PointCloud<pcl::PointXYZ>::Ptr peak_points(new pcl::PointCloud<pcl::PointXYZ>);

    boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width = recognizeOTC(point_cloud_not_cut, peak_points, "oct_dir", true);
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    std::cout<<point_cloud_not_cut->size()<<" Wow  "<< std::endl;
    viewer.showCloud(point_cloud_not_cut);
    while (!viewer.wasStopped ())
       {
       }
    std::cout<<"End";
    return 0;
}
