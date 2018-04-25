#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <algorithm>
#include <pcl/common/transforms.h>
#include "vtk_model_sampling.h"
#define NUM_FRAMES 128
#define SCALE_X 2.7
#define SCALE_Y 2.4
#define SCALE_Z 3.0


//--------------------------------------
//compute needle rotation
//--------------------------------------
Eigen::Matrix3f computeNeedleRotation(std::pair<Eigen::Vector3f, Eigen::Vector3f> direction) {
	Eigen::Vector3f zRotation = std::get<1>(direction);
	Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
	Eigen::Vector3f xRotation = up.cross(zRotation);
	xRotation.normalize();
	Eigen::Vector3f yRotation = zRotation.cross(xRotation);
	yRotation.normalize();
	Eigen::Matrix3f rotation;
	rotation << xRotation.x(), yRotation.x(), zRotation.x(),
		xRotation.y(), yRotation.y(), zRotation.y(),
		xRotation.z(), yRotation.z(), zRotation.z();
	return rotation;
}

//------------------------------------
//compute needle translation
//------------------------------------
Eigen::Vector3f computeNeedleTranslation(float tangencyPoint, Eigen::Vector3f pointOnOCTCloud, Eigen::Vector3f direction, float halfModelSize) {
	if (direction.z() < 0) {
		direction *= -1;
	}
	Eigen::Vector3f translation = pointOnOCTCloud;
	float dist = std::abs(pointOnOCTCloud.z() - tangencyPoint);
	float mult = std::abs(dist / direction.z());
	if (pointOnOCTCloud.z() < tangencyPoint) {
		translation += direction * mult;
	}
	else if (pointOnOCTCloud.z() > tangencyPoint) {
		translation -= direction * mult;
	}
	translation -= (halfModelSize / direction.z()) * direction;
	return translation;
}

//------------------------------------------------
//rotate point cloud around z axis by given angle
//------------------------------------------------
Eigen::Matrix3f rotateByAngle(float angleInDegrees, Eigen::Matrix3f currentRotation) {
	Eigen::Matrix3f rotationZ;
	Eigen::Matrix3f finalRotation = currentRotation;
	float angle = angleInDegrees * M_PI / 180.0f;
	rotationZ << std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle), 0, 0, 0, 1;
	finalRotation *= rotationZ;
	return finalRotation;
}

//---------------------------------------------------------
// compute translation given how much it should be shifted
//---------------------------------------------------------
Eigen::Vector3f shiftByValue(float shift, Eigen::Vector3f currentTranslation, Eigen::Vector3f direction) {
	Eigen::Vector3f finalTranslation = currentTranslation;
	finalTranslation += direction * (shift / direction.z());
	return finalTranslation;
}

//-----------------------------------------------------------------
// build transformation matrix from given rotation and translation
//-----------------------------------------------------------------
Eigen::Matrix4f buildTransformationMatrix(Eigen::Matrix3f rotation, Eigen::Vector3f translation) {
	Eigen::Matrix4f transformation;
	transformation.block(0, 0, 3, 3) = rotation;
	transformation.col(3).head(3) = translation;
	transformation.row(3) << 0, 0, 0, 1;
	return transformation;
}

//--------------------------------------------
//get z-rotation from transformation matrix
//--------------------------------------------
float getAngleFromMatrix(const Eigen::Matrix4f& transformation) {
	float angle = 0.0f;
	Eigen::Matrix3f end_rot = transformation.block(0, 0, 3, 3);
	Eigen::Vector3f eulerAngles = end_rot.eulerAngles(0, 1, 2);
	eulerAngles *= 180 / M_PI;
	std::cout << eulerAngles << std::endl;
	if (eulerAngles.z() < 0) {
		angle = -180 - eulerAngles.z();
	}
	else {
		angle = 180 - eulerAngles.z();
	}
	std::cout << "angle: " << angle << std::endl;
	angle *= -1.0f;
	return angle;
}

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


void convertVector3fToCl(Eigen::Vector3f vector3f, float *res) {
  res[0]=vector3f[0];
  res[1]=vector3f[1];
  res[2]=vector3f[2];
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
void convertPointXYZtoCL(pcl::PointXYZ point, float* result) {
    result[0]= point.x;
    result[1]= point.y;
    result[2]= point.z;
}

void convertMatrix3fToCL(Eigen::Matrix3f matrix3f, float* result) {
		for (int i = 0; i<3; i++) {//row
			for (int k = 0; k<3; k++) { // colm
                                result[i*3+k]=matrix3f(i,k);
			}
		}

}

void convertPointCloudToCL(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, float* res,int size) {
    for (int i = 0 ; i <size ; i++) {
        res[i*3]= pointCloud.get()->at(i).x;
        res[i*3+1]= pointCloud.get()->at(i).y;
        res[i*3+2]= pointCloud.get()->at(i).z;

    }
    std::cout<<size<< " points added"<<std::endl;
}
