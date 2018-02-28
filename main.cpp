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
#include <util.h>
#include <transformations.h>
#include <graphUtils/GraphUtils.h>
#include <CL/cl.h>
#include <pcl/common/intersections.h>

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

float computeTipX(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::pair<Eigen::Vector3f, Eigen::Vector3f> origin_and_direction_needle, float x_middle_OCT, float z_min_OCT) {
    pcl::PointXYZ min = getMinPoint(cloud);
    Eigen::VectorXf line1(6);
    line1 << x_middle_OCT, 0.0f, z_min_OCT, std::get<1>(origin_and_direction_needle)(0), 0.0f, std::get<1>(origin_and_direction_needle)(2);
    Eigen::VectorXf line2(6);
    line2 << min.x, 0.0f, min.z, std::get<1>(origin_and_direction_needle)(2), 0.0f, -std::get<1>(origin_and_direction_needle)(0);
    Eigen::Vector4f point;

    pcl::lineWithLineIntersection(line1, line2, point);
    return point.x();
}


Eigen::Matrix4f tipApproximation(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& modelTransformed,
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, std::pair<Eigen::Vector3f, Eigen::Vector3f> direction, const Eigen::Matrix4f& transformation) {
    Eigen::Matrix4f transform = transformation;
    //compute middle of OCT
    float z_min_OCT = getMinZValue(point_cloud_ptr);
    float x_middle_OCT = computeMiddle(point_cloud_ptr, z_min_OCT);

    //compute middle of CAD model
    float z_min_model = getMinZValue(modelTransformed);
    float x_middle_model = computeMiddle(modelTransformed, z_min_model);

    Eigen::Vector3f OCT_point(x_middle_OCT, 0.0f, 0.0f);
    //compute x-value which is the approximated tip
    float x_in_direction = computeTipX(modelTransformed, direction, x_middle_OCT, z_min_OCT);


    float angle_to_rotate = 0.5f;
    float sign = 1.0f;
    //rotate model until computed x-value is reached
    {
        pcl::ScopeTime t("Tip Approximation");
        float first = 0.0f;
        float second = 0.0f;
        float r = 0.0f;
        if (x_middle_model < x_in_direction) {
            sign = -1.0f;
            first = x_middle_model;
            second = x_in_direction;
        }
        else if (x_middle_model > x_in_direction) {
            sign = 1.0f;
            first = x_in_direction;
            second = x_middle_model;
        }
        //Tuan TODO : Here could be much better
        while (r < 360.0f && first < second) {
            transform = buildTransformationMatrix(rotateByAngle(sign * angle_to_rotate, transform.block(0, 0, 3, 3)), transform.block(0, 3, 3, 0));
            pcl::transformPointCloud(*model_voxelized, *modelTransformed, transform);
            if (sign < 0) {
                first = computeMiddle(modelTransformed, getMinZValue(modelTransformed));
            }
            else {
                second = computeMiddle(modelTransformed, getMinZValue(modelTransformed));
            }
            r += angle_to_rotate;
        }
    }
    return transform;
}
std::pair<Eigen::Vector3f, Eigen::Vector3f> computeNeedleDirection(pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr peak_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> inliers = getInliers(peak_points);
    for (int i = 0; i < inliers.size(); i++) {
        peak_inliers->push_back(peak_points->at(inliers.at(i)));
    }
    std::vector<Eigen::Vector3f> peak_positions;
    for (int i = 0; i < peak_inliers->points.size(); i++) {//for RANSAC use peak_inliers, else peak_points
        pcl::PointXYZ point = peak_inliers->points.at(i); //for RANSAC use peak_inliers, else peak_points
        Eigen::Vector3f eigenPoint(point.x, point.y, point.z);
        peak_positions.push_back(eigenPoint);
    }
    peak_points = peak_inliers; //only when using RANSAC
    return fitLine(peak_positions);
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

//TODO:
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
    std::vector<float> result{point.x, point.y,point.z};
    return result;
}

std::vector<float> convertMatrix3fToCL(Eigen::Matrix3f matrix3f) {
  std::vector<float> vector(matrix3f.data(),matrix3f.data()+ matrix3f.cols()*matrix3f.rows());
  return vector;
}

std::vector<std::vector<float>> convertPointCloudToCL(pcl::PointCloud<pcl::PointXYZ> pointCloud) {
  std::vector<std::vector<float>> pointCloudVec;
  int size = pointCloud.size();
  for (int i = 0 ; i <size ; i++) {
    pointCloudVec.push_back(convertPointXYZtoCL(pointCloud.at(i)));
  }
  return pointCloudVec;
}

//

int main()
{
    //TODO : Parameterize this please
    std::string path= "/home/tuan/Desktop/Back up/BA/Source Code/Models";
    std::string oct_dir ="/home/tuan/Desktop/Back up/BA/Source Code/oct";

    //http://pointclouds.org/documentation/tutorials/cloud_viewer.php
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>); //Create pointer to a points cloud of PointXYZ Type (3 Axis x,y,z)
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_not_cut(new pcl::PointCloud<pcl::PointXYZ>);  //Create pointer to a points cloud of PointXYZ Type (3 Axis x,y,z)
    pcl::PointCloud<pcl::PointXYZ>::Ptr peak_points(new pcl::PointCloud<pcl::PointXYZ>);

    boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width = recognizeOTC(point_cloud_not_cut, peak_points, "oct_dir", true);
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    /*
    viewer.showCloud(point_cloud_not_cut);
    while (!viewer.wasStopped ())   {
    }
    */
    //Copied Code
    std::cout<<"End";
    pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized(new pcl::PointCloud<pcl::PointXYZ>());
    std::cout<<"Hey here is fine"<<std::endl;

    generatePointCloudFromModel(modelCloud, model_voxelized, path);

    cutPartOfModel(point_cloud_not_cut, point_cloud_ptr, getModelSize(model_voxelized) - 0.1f + getMinZValue(point_cloud_not_cut));
    std::pair<Eigen::Vector3f, Eigen::Vector3f> direction = computeNeedleDirection(peak_points);
    std::cout << "origin: " << std::endl << std::get<0>(direction) << std::endl << "direction: " << std::endl << std::get<1>(direction) << std::endl;

    Eigen::Matrix3f rotation = computeNeedleRotation(direction);

    //WTF is this shit
    Eigen::Vector3f euler = rotation.eulerAngles(0, 1, 2) * 180 / M_PI;
    rotation = rotateByAngle(180 - euler.z(), rotation);
    std::cout << "euler angles: " << std::endl << rotation.eulerAngles(0, 1, 2) * 180 / M_PI << std::endl;

    float tangencyPoint = regression(needle_width) / (float)NUM_FRAMES * SCALE_Z; //scaling
    std::cout << "tangency point: " << tangencyPoint << std::endl;

    Eigen::Vector3f initialTranslation = computeNeedleTranslation(tangencyPoint, std::get<0>(direction),std::get<1>(direction), getModelSize(model_voxelized) / 2);
    std::cout << "translation: " << std::endl << initialTranslation << std::endl;

    Eigen::Matrix4f transformation = buildTransformationMatrix(rotation, initialTranslation);

    pcl::PointCloud<pcl::PointXYZ>::Ptr modelTransformed(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud(*model_voxelized, *modelTransformed, transformation);

    transformation = tipApproximation(point_cloud_ptr, modelTransformed, model_voxelized, direction, transformation);

    float end_angle = getAngleFromMatrix(transformation);

    std::vector<std::tuple<float, float, float>> correspondence_count;
    //angle and count
    std::vector<std::pair<float, float>> angle_count;
    //shift and count
    std::vector<std::pair<float, float>> shift_count;

    std::vector<std::tuple<float, float, float>> correspondence_count2;
    //angle and count
    std::vector<std::pair<float, float>> angle_count2;
    //shift and count
    std::vector<std::pair<float, float>> shift_count2;

    std::vector<std::tuple<float, float, float>> correspondence_count3;
    //angle and count
    std::vector<std::pair<float, float>> angle_count3;
    //shift and count
    std::vector<std::pair<float, float>> shift_count3;

    std::vector<std::tuple<float, float, float>> correspondence_count4;
    //angle and count
    std::vector<std::pair<float, float>> angle_count4;
    //shift and count
    std::vector<std::pair<float, float>> shift_count4;
    //initialize interval values -90,90

    float angleStart = end_angle - 5.0f;
    float angleEnd = end_angle + 5.0f;
    float angleStep = 1.0f;
    float shiftStart = 0.0f;
    float shiftEnd = 0.5;
    float shiftStep = 0.05f;
    //more initialization
    int max_index_angles = 0;
    int max_index_shift = 0;
    int correspondence_index = 0;
    float max_angle = 0.0f;
    float max_shift = 0.0f;
    //STAND :: FROM NOW ON OPENCL



    //OPENCL PART:

    //https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj , resobj = NULL;

    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    std::vector<int> input {1,2,3,4,5,6,7,8,9,10};

    int length = 10 ;


    FILE *fp;
    char fileName[] = "/home/tuan/Desktop/OpenCLBA-Local/OpenCLBA-Prod/hello.cl";
    char *source_str;
    size_t source_size;

    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
    }
    source_str = (char*)malloc(0x100000);
    source_size = fread(source_str,1,0x100000, fp);
    fclose(fp);

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    std::cout<<ret<<" code"<<std::endl;

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    std::cout<<ret<<" code"<<std::endl;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    std::cout<<ret<<" code"<<std::endl;

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    //Check Concept of memory
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,length * sizeof(int), NULL, &ret);
    std::cout<<ret<<" code"<<std::endl;


    resobj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, length * sizeof(int), NULL, &ret);
    std::cout<<ret<<" code"<<std::endl;


    program = clCreateProgramWithSource(context,1,(const char**)&source_str, (const size_t*)&source_size, &ret);
    std::cout<<ret<<" code"<<std::endl;

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    std::cout<<ret<<" code"<<std::endl;

    kernel = clCreateKernel(program, "hello", &ret);
    std::cout<<ret<<" code"<<std::endl;

    ret = clSetKernelArg(kernel,0, sizeof(memobj),(void *)&memobj);
    std::cout<<ret<<" code"<<std::endl;

    ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
    std::cout<<ret<<" code"<<std::endl;

    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, 10 * sizeof(int),&input[0], 0, NULL, NULL);


    //TODO :

    cl_mem correspondence_count_mem = clCreateBuffer(context,CL_MEM_READ_WRITE, 1000, NULL, &ret);

    cl_mem angle_count_mem = clCreateBuffer(context,CL_MEM_READ_WRITE, 1000, NULL, &ret);

    cl_mem shift_count_mem = clCreateBuffer(context,CL_MEM_READ_WRITE, 1000, NULL, &ret);

    //TODO : 4 big work Group












    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    for (int i = 0 ; i <10 ; i++) {
        std::cout<<input[i]<<" "<<std::endl;
    }









    return 0;
}
