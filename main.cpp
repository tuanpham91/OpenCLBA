#define CL_HPP_TARGET_OPENCL_VERSION 200
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
cl_mem memobj , resobj, argsMemObj, countMemobj, initialTranslationMemObj, directionMemObj, modelVoxelizedMembObj, pointCloudPtrMemObj, rotationMemObj, correspondenceResultMemObj=NULL;

cl_program  program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;

int determinNumWorkItems(int sizeOfProblem) {
    return ((sizeOfProblem+63)/64)*64;
}

void printDeviceInfoWorkSize(cl_device_id device) {
    size_t size;
    size_t worksizes[3];
    clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(size_t)*3,worksizes,NULL);
    std::cout<< " Work sizes are  :" <<worksizes[0]<<" ,"<<worksizes[1]<<" ,"<<worksizes[2]<<std::endl;
}

void shift_and_roll_without_sum_in_cl(float angle_min, float angle_max, float angle_step,
                                      float shift_min, float shift_max, float shift_step,
                                      std::vector<std::tuple<float, float, float>>& count,
                                      Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, int number_of_points_per_work_item
                                      ) {
    clock_t begin = clock();
    clock_t end = clock();
    double elapsed_secs1 = double(end - begin) / CLOCKS_PER_SEC;

    FILE *fp;
    char fileName[] = "/home/tuan/Desktop/OpenCLBA-Local/OpenCLBA/kernel-original.cl";
    char *source_str;
    size_t source_size;
    cl_int ret;

    int num_angle_steps_s = std::round((angle_max - angle_min) / angle_step) + 1;
    int num_shift_steps_s = std::round((shift_max - shift_min) / shift_step) + 1;

    std::cout << "Number of should be  dimension size:  " << num_angle_steps_s<< " " <<num_shift_steps_s<< std::endl;

    fp = fopen(fileName, "r");
    if (!fp) {
    fprintf(stderr, "Failed to load kernel\n");
    exit(1);
    }

    source_str = (char*)malloc(0x100000);
    source_size = fread(source_str,1,0x100000, fp);
    fclose(fp);

    int num_angle_steps= std::round((angle_max - angle_min) / angle_step) + 1;
    int num_shift_steps = std::round((shift_max - shift_min) / shift_step) + 1;
    int prod = num_angle_steps*num_shift_steps;

    //Why allocate memory ? see here:https://stackoverflow.com/questions/10575544/difference-between-array-type-and-array-allocated-with-malloc

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    std::cout<<ret<<" 1. code"<<std::endl;

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    std::cout<<ret<<" 2.  code"<<std::endl;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    std::cout<<ret<<" 3.  code"<<std::endl;
    cl_queue_properties props[] = {
      CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
      0
    };
    std::cout<<ret<<" 3.1  code"<<std::endl;

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    std::cout<<ret<<" 3.1  code"<<std::endl;


    //Check Concept of memory


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
    end = clock() ;
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for Build programm method is : " <<elapsed_secs<<std::endl;

    printDeviceInfoWorkSize(device_id);
    kernel = clCreateKernel(program,"transforming_models", &ret);

    float args[21] ={angle_min, angle_max, angle_step, shift_min, shift_max, shift_step,initialTranslation[0],initialTranslation[1],initialTranslation[2],direction[0],direction[1],direction[2],rotation[0],rotation[1],rotation[2],rotation[3],rotation[4],rotation[5],rotation[6],rotation[7],rotation[8]};
    //0. Arg
    argsMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR ,21*sizeof(float),args,&ret);
    ret = clSetKernelArg(kernel,0, sizeof(argsMemObj),(void *)&argsMemObj);

    //4. Arg model_voxelized
    int model_voxelized_as_array_size = static_cast<int>(model_voxelized.get()->size())*3;
    std::cout<<"DEBUG : Size of model voxelized : " << model_voxelized_as_array_size<< "or " << model_voxelized.get()->size()<< "last point is " << model_voxelized.get()->at(6778).x<< " " <<model_voxelized.get()->at(6778).y<< "  "<<model_voxelized.get()->at(6778).z <<std::endl;
    float *model_voxelized_as_array = new float[model_voxelized_as_array_size]();
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array,model_voxelized_as_array_size/3);
    modelVoxelizedMembObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , sizeof(float)*model_voxelized_as_array_size,model_voxelized_as_array,&ret);
    //std::cout<<ret<<" Arg code 4.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,1,sizeof(modelVoxelizedMembObj), &modelVoxelizedMembObj);
    //std::cout<<ret<<" Arg code 4.2:"<<std::endl;


    //9. Work size dimension
    cl_mem workSizeMemObj = NULL;
    int* worksizes = new int[3]();
    worksizes[0]= num_angle_steps;
    worksizes[1]= num_shift_steps;
    worksizes[2]= static_cast<int>(model_voxelized->size());
    workSizeMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*2,worksizes,&ret);
    std::cout<<ret<< " Arg code 9.1 :"<<std::endl;
    ret=clSetKernelArg(kernel,2,sizeof(workSizeMemObj),&workSizeMemObj);
    std::cout<<ret<< " Arg code 9.2 :"<<std::endl;



    //12. input_transformed
    cl_mem inputTransformedMemObj =NULL;
    int size_input_transformed_array =worksizes[2]*3*num_angle_steps*num_shift_steps;
    float* input_transformed_as_array = new float[size_input_transformed_array]();
    //std::cout<<"DEBUG : Size of input transformed array :" << size_input_transformed_array << std::endl;
    inputTransformedMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,sizeof(float)*size_input_transformed_array,input_transformed_as_array,&ret);
    //std::cout<<ret<< " Arg code 11.1 :"<<std::endl;
    ret= clSetKernelArg(kernel,3,sizeof(inputTransformedMemObj),&inputTransformedMemObj);
    //std::cout<<ret<< " Arg code 12.2:"<<std::endl;

    size_t work_units[3] ={(size_t)num_angle_steps_s,(size_t)num_shift_steps_s, model_voxelized.get()->size()};

    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 3 , NULL,work_units, NULL, 0, NULL, NULL);
    std::cout<<"Running Program, code:" << ret <<std::endl;

    clFlush(command_queue);
    clFinish(command_queue);

    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for 1. kernel method is : " <<elapsed_secs<<std::endl;
    std::cout<<"DEBUG : Last elements of input transformed is " << input_transformed_as_array[2460774]<< " "<<input_transformed_as_array[2460775]<< " "<<input_transformed_as_array[2460776]<< std::endl;


    kernel = clCreateKernel(program,"find_correspondences", &ret);
    std::cout<<ret<<" Part 2 :find correspondence "<<std::endl;
    int number_of_points_to_calculate = worksizes[2]*num_angle_steps*num_shift_steps;
    int int_args[2] = {1,number_of_points_to_calculate};
    cl_mem intArgs = NULL;
    intArgs = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int)*2,int_args,&ret);
    //std::cout<<ret<<" Part 2.1.0 : "<<std::endl;
    ret = clSetKernelArg(kernel,0,sizeof(intArgs), &intArgs);
    //std::cout<<ret<<" Part 2.1.0 : "<<std::endl;

    size_t work_units2[1]= {(size_t)determinNumWorkItems(number_of_points_to_calculate)};

    int point_cloud_ptr_array_size = static_cast<int>(point_cloud_ptr.get()->size())*3;
    float* point_cloud_ptr_as_array = new float[point_cloud_ptr_array_size]();
    convertPointCloudToCL(point_cloud_ptr,point_cloud_ptr_as_array,point_cloud_ptr_array_size/3);
    std::cout<< "Size of pointCloud array is " << point_cloud_ptr_array_size<< " points , last value is: " << point_cloud_ptr_as_array[point_cloud_ptr_array_size-1]<< " compare with "<< point_cloud_ptr.get()->at(point_cloud_ptr_array_size/3-1).z <<std::endl;
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*point_cloud_ptr_array_size, point_cloud_ptr_as_array,&ret);
    //std::cout<<ret<<" Part 2.1.2 : "<<std::endl;
    ret = clSetKernelArg(kernel,1,sizeof(pointCloudPtrMemObj),&pointCloudPtrMemObj);
    //std::cout<<ret<<" Part 2.1.2 : "<<std::endl;

    cl_mem correspondenceRes= NULL;
    int size_correspondence_result = static_cast<int>(model_voxelized->size())*3*num_angle_steps*num_shift_steps;
    std::cout<<"DEBUG : Number of max correspondence found:"<<size_correspondence_result<<std::endl;
    float* correspondence_result = new float[size_correspondence_result]();
    correspondenceRes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size_correspondence_result,correspondence_result,&ret);
    //std::cout<<ret<<" Part 2.1.3 : "<<std::endl;
    ret = clSetKernelArg(kernel,2,sizeof(correspondenceRes), &correspondenceRes);
    //std::cout<<ret<<" Part 2.1.3 : "<<std::endl;

    //4. Arg correspondence_result_count;
    /*
    cl_mem corr_result = NULL;
    int corr_result_size = number_of_points_to_calculate;
    //int *corr_result_count = new int[corr_result_size]();
    float *corr_result_count = new float[corr_result_size]();
    corr_result= clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(float)*corr_result_size,corr_result_count,&ret);
    std::cout<<ret<<" Part 2.1.4 : "<<std::endl;
    ret=clSetKernelArg(kernel,3,sizeof(corr_result),&corr_result);
    std::cout<<ret<<" Part 2.1.4 : "<<std::endl;
    std::cout<<"DEBUG : Number of max Work Items :"<<corr_result_size<< " "<<std::endl;
    */

    cl_mem sourceSizesMemObj = NULL;
    int *sources_sizes= new int[2]();
    sources_sizes[0]= static_cast<int>(model_voxelized->size());
    sources_sizes[1]= static_cast<int>(point_cloud_ptr_array_size/3);
    //std::cout<<"DEBUG : Sources size is Voxelized : " << sources_sizes[0]<< " Point Cloud : "<< sources_sizes[1]<<std::endl;
    sourceSizesMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*2, sources_sizes, &ret);
    //std::cout<<ret<<" Part 2.1.6 : "<<std::endl;
    ret = clSetKernelArg(kernel,3,sizeof(sourceSizesMemObj),&sourceSizesMemObj);
   // std::cout<<ret<<" Part 2.1.6 : "<<std::endl;

    ret= clSetKernelArg(kernel,4,sizeof(inputTransformedMemObj),&inputTransformedMemObj);
    //std::cout<<ret<<" Part 2.1.7 : "<<std::endl;

    size_t local_work_size[1] = {(size_t) 64};
    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,work_units2, local_work_size, 0, NULL, NULL);
    std::cout<<"Running Program part 2, code:" << ret <<std::endl;

    clFlush(command_queue);
    clFinish(command_queue);
    /* FOR SPEED REASON
    ret = clEnqueueReadBuffer(command_queue,corr_result,CL_TRUE,0,sizeof(float)*corr_result_size, &corr_result_count[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;
    */
    /* FOR SPEED REASON
    ret = clEnqueueReadBuffer(command_queue,correspondenceRes,CL_TRUE,0,sizeof(float)*size_correspondence_result, &correspondence_result[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;
    */
    //for ( int i = corr_result_size-100; i <corr_result_size; i++) {
    /*FOR SPEED REASON
    for ( int i = 0; i <100; i++) {
       //std::cout << corr_result_count[i]<<"  ";
        //std::cout << corr_result_count[i]<<"  ";
        std::cout<<correspondence_result[i]<<"  ";
    }
    */

    clock_t end2 = clock() ;
    elapsed_secs = double(end2 - end) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for 2. kernel method is : " <<elapsed_secs<<std::endl;



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

    float angleStart = end_angle - 5.0f;
    float angleEnd = end_angle + 5.0f;
    float angleStep = 1.0f;
    float shiftStart = 0.0f;
    float shiftEnd = 0.5;
    float shiftStep = 0.05f;

    int max_index_angles = 0;
    int max_index_shift = 0;
    int correspondence_index = 0;
    float max_angle = 0.0f;
    float max_shift = 0.0f;


    //https://streamhpc.com/blog/2013-04-28/opencl-error-codes /

    //https://stackoverflow.com/questions/26804153/opencl-work-group-concept

    //http://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html

    shift_and_roll_without_sum_in_cl(angleStart,angleEnd, angleStep,shiftStart, shiftEnd, shiftStep,
                                     correspondence_count, rotation,
                                     initialTranslation, std::get<1>(direction), model_voxelized,
                                     point_cloud_ptr,5);


    return 0;
}
