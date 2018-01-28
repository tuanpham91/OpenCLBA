#include <iostream>

#include <CL/cl.h>
#include <pcl/cloud_iterator.h>


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

std::vector<std::vector<float>> convertMatrix4fToCL(Eigen::Matrix4f matrix4f) {
  std::vector<std::vector<float>> matrixVectors;
  std::vector<float> column1 = matrix3f.col(0);
  matrixVectors.push_back(column1);
  std::vector<float> column2 = matrix3f.col(1);
  matrixVectors.push_back(column2);
  std::vector<float> column3 = matrix3f.col(2);
  matrixVectors.push_back(column3);
  std::vector<float> column4 = matrix4f.col(3);
  matrixVectors.push_back(column4);
  return matrixVectors;

}



std::vector<std::vector<float>> convertMatrix3fToCL(Eigen::Matrix3f matrix3f) {
  std::vector<std::vector<float>> matrixVectors;
  std::vector<float> column1 = matrix3f.col(0);
  matrixVectors.push_back(column1);
  std::vector<float> column2 = matrix3f.col(1);
  matrixVectors.push_back(column2);
  std::vector<float> column3 = matrix3f.col(2);
  matrixVectors.push_back(column3);
  return matrixVectors;
}

std::vector<std::vector<float>> convertPointCloudToCL(pcl::PointCloud<pcl::PointXYZ> pointCloud) {
  std::vector<std::vector<float>> pointCloud;
  int size = pointCloud.size();
  for (int i = 0 ; i <size ; i++) {
    pointCloud.push_back(convertPointXYZtoCL(pointCloud.get(i)));
  }
}

std::vector<float> convertPointXYZtoCL(pcl::PointXYZ point) {
  std::vector<float> result{point.x, point.y,point.z);
  return result;
}

int main()
{
    //TODO : shift_and_roll_without_sum (OpenCL Task1)

    cout << "Hello World!" << endl;
    return 0;
}
