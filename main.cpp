#include <opencv2/opencv.hpp>

#include <pcl/pcl_macros.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/common/time.h>
#include <pcl/common/intersections.h>
#include <pcl/filters/voxel_grid.h>


#include <tuple>

#include "util.h"
#include "regression.h"
#include "graphUtils/GraphUtils.h"



#include <CL/cl2.hpp>
#include "misc.h"
/* Programm to replicate OPENCV part, without shifting algorithmus */
using namespace std;
#define NUM_FRAMES 128
#define SCALE_X 2.7
#define SCALE_Y 2.4
#define SCALE_Z 3.0



cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj , resobj, argsMemObj, countMemobj, initialTranslationMemObj, directionMemObj, modelVoxelizedMembObj, pointCloudPtrMemObj, rotationMemObj=NULL;

cl_program program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;


void shift_and_roll_without_sum_in_cl(float angle_min, float angle_max, float angle_step,
                                      float shift_min, float shift_max, float shift_step,
                                      std::vector<std::tuple<float, float, float>>& count,
                                      Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr
                                      ) {

    FILE *fp;
    char fileName[] = "/home/tuan/Desktop/OpenCLBA-Local/OpenCLBA/kernel-original.cl";
    char *source_str;
    size_t source_size;
    cl_int ret;

    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
    }

    source_str = (char*)malloc(0x100000);
    source_size = fread(source_str,1,0x100000, fp);
    fclose(fp);

    int num_angle_steps = std::round((angle_max - angle_min) / angle_step) + 1;
    int num_shift_steps = std::round((shift_max - shift_min) / shift_step) + 1;
    int prod = num_angle_steps*num_shift_steps;

    //Why allocate memory ?  see here : https://stackoverflow.com/questions/10575544/difference-between-array-type-and-array-allocated-with-malloc

    float count_cl [prod][3];

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    std::cout<<ret<<" code"<<std::endl;

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    std::cout<<ret<<" code"<<std::endl;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    std::cout<<ret<<" code"<<std::endl;

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);


    //Check Concept of memory
    float args[6] ={angle_min, angle_max, angle_step, shift_min, shift_max, shift_step};

    program = clCreateProgramWithSource(context,1,(const char**)&source_str, (const size_t*)&source_size, &ret);
    std::cout<<ret<<" code"<<std::endl;

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    std::cout<<ret<<" code"<<std::endl;
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    // TODO : Adjust kernel here, 4 //  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_one); 4.ARG!
    kernel = clCreateKernel(program,"shift_and_roll_without_sum_loop", &ret);
    std::cout<<ret<<" Arg code 0: "<<std::endl;

    //0. Arg
    argsMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE,6*sizeof(float),args,&ret);
    ret = clSetKernelArg(kernel,0, sizeof(argsMemObj),(void *)&argsMemObj);
    std::cout<<ret<<" Arg code 1 :"<<std::endl;

    //1. Arg count
    countMemobj = clCreateBuffer(context,CL_MEM_READ_WRITE,prod*sizeof(float),&count,&ret);
    ret = clSetKernelArg(kernel,1 , sizeof(countMemobj), (void *)&countMemobj);
    std::cout<<ret<<" Arg code 2 :"<<std::endl;

    //2. Arg initialTranslation
    initialTranslationMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE,3*sizeof(float),initialTranslation.data(),&ret);
    ret = clSetKernelArg(kernel,2,sizeof(initialTranslation), &initialTranslationMemObj);
    std::cout<<ret<<" Arg code 3 :"<<std::endl;

    //3. Arg direction

    directionMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*sizeof(float),direction.data(),&ret);
    ret = clSetKernelArg(kernel,3,sizeof(direction), &directionMemObj);
    std::cout<<ret<<" Arg code 4 :"<<std::endl;

    //4. Arg model_voxelized
    float* model_voxelized_as_array = new float[model_voxelized.get()->size()*3];
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array);
    modelVoxelizedMembObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(model_voxelized_as_array),model_voxelized_as_array,&ret);
    ret = clSetKernelArg(kernel,4,sizeof(model_voxelized), &modelVoxelizedMembObj);
    std::cout<<ret<<" Arg code 5 :"<<std::endl;

    //5.Arg point_cloud_ptr
    float* point_cloud_ptr_as_array = new float[point_cloud_ptr.get()->size()*3];
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array);
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(point_cloud_ptr),point_cloud_ptr_as_array,&ret);
    ret = clSetKernelArg(kernel,5,sizeof(point_cloud_ptr), &pointCloudPtrMemObj);
    std::cout<<ret<<" Arg code 6 :"<<std::endl;

    //6.Arg rotation
    rotationMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(rotation.data()),rotation.data(),&ret);
    ret = clSetKernelArg(kernel,6, sizeof(rotation), &rotationMemObj);
    std::cout<<ret<<"Arg code 7 :"<<std::endl;



}


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
    /*
    viewer.showCloud(point_cloud_not_cut);
    while (!viewer.wasStopped ())   {
    }
    */
    //Copied Code

    pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized(new pcl::PointCloud<pcl::PointXYZ>());

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
    std::cout<<"DEBUG"<<std::endl;
    //PROBLEM HERE
    transformation = tipApproximation(point_cloud_ptr, modelTransformed, model_voxelized, direction, transformation);

    float end_angle = getAngleFromMatrix(transformation);
    std::cout<<"DEBUG"<<std::endl;

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


    std::cout<<"DEBUG"<<std::endl;
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

    cl_int ret;
    std::vector<int> input {1,2,3,4,5,6,7,8,9,10};

    int length = 10 ;


    FILE *fp;
    char fileName[] = "/home/tuan/Desktop/OpenCLBA-Local/OpenCLBA/kernel.cl";
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

    //https://stackoverflow.com/questions/26804153/opencl-work-group-concept



    //http://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html

    //Test with one iteration

    shift_and_roll_without_sum_in_cl(angleStart,angleEnd, angleStep,shiftStart, shiftEnd, shiftStep,
                                     correspondence_count, rotation,
                                     initialTranslation, std::get<1>(direction), model_voxelized,
                                     point_cloud_ptr);









    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);




    return 0;
}
