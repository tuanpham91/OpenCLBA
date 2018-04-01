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

#include <ctime>
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
size_t work_units =4;

void shift_and_roll_without_sum_in_cl(float angle_min, float angle_max, float angle_step,
                                      float shift_min, float shift_max, float shift_step,
                                      std::vector<std::tuple<float, float, float>>& count,
                                      Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr
                                      ) {
    clock_t begin = clock();

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


    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    std::cout<<ret<<" 1.  code"<<std::endl;

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    std::cout<<ret<<" 2.  code"<<std::endl;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    std::cout<<ret<<" 3.  code"<<std::endl;

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);


    //Check Concept of memory
    float args[6] ={angle_min, angle_max, angle_step, shift_min, shift_max, shift_step};

    program = clCreateProgramWithSource(context,1,(const char**)&source_str, (const size_t*)&source_size, &ret);
    std::cout<<ret<<" 4.  code"<<std::endl;

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    std::cout<<ret<<" 5. code"<<std::endl;
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


    std::cout<<"START KERNEL"<<std::endl;
    kernel = clCreateKernel(program,"shiftAndRollWithoutSumLoop", &ret);
    std::cout<<ret<<" Arg code 0: "<<std::endl;

    //0. Arg
    argsMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE,6*sizeof(float),args,&ret);
    ret = clSetKernelArg(kernel,0, sizeof(argsMemObj),(void *)&argsMemObj);
    std::cout<<ret<<" Arg code 1 :"<<std::endl;

    /*
    //1. Arg count
    countMemobj = clCreateBuffer(context,CL_MEM_READ_WRITE,prod*sizeof(float),&count,&ret);
    ret = clSetKernelArg(kernel,1 , sizeof(countMemobj), (void *)&countMemobj);
    std::cout<<ret<<" Arg code 2 :"<<std::endl;
    */

    //2. Arg initialTranslation
    float initialTranslationData[3];
    convertVector3fToCl(initialTranslation,initialTranslationData);
    initialTranslationMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE,3*sizeof(float),initialTranslationData,&ret);
    ret = clSetKernelArg(kernel,2,sizeof(initialTranslationMemObj), &initialTranslationMemObj);
    std::cout<<ret<<" Arg code 3 :"<<std::endl;

    //3. Arg direction

    float directionData[3];
    convertVector3fToCl(initialTranslation,initialTranslationData);
    directionMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*sizeof(float),direction.data(),&ret);
    ret = clSetKernelArg(kernel,3,sizeof(directionMemObj), &directionMemObj);
    std::cout<<ret<<" Arg code 4 :"<<std::endl;

    //4. Arg model_voxelized
    float* model_voxelized_as_array = new float[model_voxelized.get()->size()*3];
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array);
    modelVoxelizedMembObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(model_voxelized_as_array),model_voxelized_as_array,&ret);
    ret = clSetKernelArg(kernel,4,sizeof(modelVoxelizedMembObj), &modelVoxelizedMembObj);
    std::cout<<ret<<" Arg code 5 :"<<std::endl;

    //5.Arg point_cloud_ptr
    int size_correspondence_count = point_cloud_ptr.get()->size()*3;
    float* correspondence_count = new float[size_correspondence_count];
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array);
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(correspondence_count),correspondence_count,&ret);
    ret = clSetKernelArg(kernel,1,sizeof(pointCloudPtrMemObj), &pointCloudPtrMemObj);
    std::cout<<ret<<" Arg code 6 :"<<std::endl;

    //6.Arg rotation
    float* rotation_as_array = new float[9];
    convertMatrix3fToCL(rotation,rotation_as_array);
    rotationMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(rotation_as_array),rotation_as_array,&ret);
    ret = clSetKernelArg(kernel,5, sizeof(rotationMemObj), &rotationMemObj);
    std::cout<<ret<<" Arg code 7 :"<<std::endl;

    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,&work_units, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(command_queue,pointCloudPtrMemObj,CL_TRUE,0,sizeof(correspondence_count), correspondence_count,0,NULL,NULL);
    clock_t end = clock();
    std::cout<< (double)(end-begin)/CLOCKS_PER_SEC <<std::endl;
    //TODO : Recheck Kernel and Args

    //point_cloud_ptr_as_array is a vector of tupel <float, float, float> actually
    //free memory





    kernel = clCreateKernel(program,"computeDifferencesForCorrespondence", &ret);
    std::cout<<ret<<" Part 2.: "<<std::endl;

    //TODO : Figure out size of correspondence Count
    cl_mem args_size_mem_obj = NULL;
    int *correspondence_count_real_size = new int( point_cloud_ptr->size());



    //TODO : Size of angle_count, shift_count is right the size of correspondent_count -> Must find size of correspondence_count
    float *angle_count = new float[point_cloud_ptr->size()*2];
    float *shift_count = new float[point_cloud_ptr->size()*2];

    args_size_mem_obj = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(int),correspondence_count_real_size,&ret);
    std::cout<<ret<<" Part 2.1.1 : "<<std::endl;
    ret = clSetKernelArg(kernel,1,sizeof(args_size_mem_obj), &args_size_mem_obj);
    std::cout<<ret<<" Part 2.1.2 : "<<std::endl;

    pointCloudPtrMemObj = NULL ;
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(correspondence_count),correspondence_count,&ret);
    std::cout<<ret<<" Part 2.2.1 : "<<std::endl;
    ret = clSetKernelArg(kernel,0,sizeof(pointCloudPtrMemObj), &pointCloudPtrMemObj);
    std::cout<<ret<<" Part 2.2.2 : "<<std::endl;


    cl_mem shift_count_mem_obj = NULL;
    shift_count_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(shift_count),shift_count,&ret);
    std::cout<<ret<<" Part 2.3.1 : "<<std::endl;
    ret = clSetKernelArg(kernel,3,sizeof(shift_count_mem_obj), &shift_count_mem_obj);
    std::cout<<ret<<" Part 2.3.2 : "<<std::endl;


    cl_mem angle_count_mem_obj = NULL;
    angle_count_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(angle_count),angle_count,&ret);
    std::cout<<ret<<" Part 2.4.1 : "<<std::endl;
    ret = clSetKernelArg(kernel,0,sizeof(angle_count_mem_obj), &angle_count_mem_obj);
    std::cout<<ret<<" Part 2.4.2 : "<<std::endl;


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

    transformation = tipApproximation(point_cloud_ptr, modelTransformed, model_voxelized, direction, transformation);

    float end_angle = getAngleFromMatrix(transformation);

    std::vector<std::tuple<float, float, float>> correspondence_count;
    //angle and count
    std::vector<std::pair<float, float>> angle_count;
    //shift and count
    std::vector<std::pair<float, float>> shift_count;

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



    //TODO : 4 big work Group

    //https://stackoverflow.com/questions/26804153/opencl-work-group-concept



    //http://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html

    //Test with one iteration

    shift_and_roll_without_sum_in_cl(angleStart,angleEnd, angleStep,shiftStart, shiftEnd, shiftStep,
                                     correspondence_count, rotation,
                                     initialTranslation, std::get<1>(direction), model_voxelized,
                                     point_cloud_ptr);



    return 0;
}
