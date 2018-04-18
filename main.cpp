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

cl_program program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;


void printDeviceInfoWorkSize(cl_device_id device) {
    size_t size;
    size_t worksizes[3];
    clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(size_t)*3,worksizes,NULL);
    std::cout<< " Work sizes are " <<worksizes[0]<<" ,"<<worksizes[1]<<" ,"<<worksizes[2]<<std::endl;
}

void shift_and_roll_without_sum_in_cl(float angle_min, float angle_max, float angle_step,
                                      float shift_min, float shift_max, float shift_step,
                                      std::vector<std::tuple<float, float, float>>& count,
                                      Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr
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
    size_t work_units[2] ={(size_t)num_angle_steps_s,(size_t)num_shift_steps_s};

    std::cout << "Number of  should be dimension size: " << num_angle_steps_s<< " " <<num_shift_steps_s<< std::endl;


    /* Load the source code containing the k ernel*/
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

    //Why allocate memory ?  see here:https://stackoverflow.com/questions/10575544/difference-between-array-type-and-array-allocated-with-malloc


    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    std::cout<<ret<<" 1.  code"<<std::endl;

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    std::cout<<ret<<" 2.  code"<<std::endl;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    std::cout<<ret<<" 3.  code"<<std::endl;
    cl_queue_properties props[] = {
      CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
      0
    };
    std::cout<<ret<<" 3.1 code"<<std::endl;

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    std::cout<<ret<<" 3.1 code"<<std::endl;


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

    printDeviceInfoWorkSize(device_id);
    kernel = clCreateKernel(program,"shiftAndRollWithoutSumLoop", &ret);

    //0. Arg
    argsMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR ,6*sizeof(float),args,&ret);
    std::cout<<ret<<" Arg code 1.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,0, sizeof(argsMemObj),(void *)&argsMemObj);
    std::cout<<ret<<" Arg code 1.2 :"<<std::endl;


    //2. Arg initialTranslation
    float initialTranslationData[3];
    convertVector3fToCl(initialTranslation,initialTranslationData);
    initialTranslationMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR ,3*sizeof(float),initialTranslationData,&ret);
    std::cout<<ret<<" Arg code 2.1 "<<std::endl;
    ret = clSetKernelArg(kernel,1,sizeof(initialTranslationMemObj), &initialTranslationMemObj);
    std::cout<<ret<<" Arg code 2.2:"<<std::endl;

    //3. Arg direction

    float directionData[3];
    convertVector3fToCl(initialTranslation,initialTranslationData);
    directionMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 3*sizeof(float),direction.data(),&ret);
    std::cout<<ret<<" Arg code 3.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,2,sizeof(directionMemObj), &directionMemObj);
    std::cout<<ret<<" Arg code 3.2 :"<<std::endl;

    //4. Arg model_voxelized
    int model_voxelized_as_array_size = model_voxelized.get()->size()*3;
    float* model_voxelized_as_array = new float[model_voxelized_as_array_size]();
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array);
    modelVoxelizedMembObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , sizeof(float)*model_voxelized_as_array_size,model_voxelized_as_array,&ret);
    std::cout<<ret<<" Arg code 4.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,3,sizeof(modelVoxelizedMembObj), &modelVoxelizedMembObj);
    std::cout<<ret<<" Arg code 4.2 :"<<std::endl;

    //5.Arg point_cloud_ptr
    int point_cloud_ptr_array_size = point_cloud_ptr.get()->size()*3;
    float* point_cloud_ptr_as_array = new float[point_cloud_ptr_array_size];
    std::cout<< "Size of pointCloud is " << static_cast<int>((point_cloud_ptr.get()->size()))<<std::endl;
    convertPointCloudToCL(point_cloud_ptr,point_cloud_ptr_as_array);
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*point_cloud_ptr_array_size, point_cloud_ptr_as_array,&ret);
    std::cout<<ret<<" Arg code 5.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,4,sizeof(pointCloudPtrMemObj),&pointCloudPtrMemObj);
    std::cout<<ret<<" Arg code 5.2 :"<<std::endl;

     //6.Arg rotation
    float* rotation_as_array = new float[9];
    convertMatrix3fToCL(rotation,rotation_as_array);
    rotationMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*9,rotation_as_array,&ret);
    std::cout<<ret<<" Arg code 6.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,5, sizeof(rotationMemObj), &rotationMemObj);
    std::cout<<ret<<" Arg code 6.2 :"<<std::endl;

    //7. Arg correspondence_result
    cl_mem correspondenceRes= NULL;
    int size_correspondence_result = model_voxelized->size()*3*num_angle_steps*num_shift_steps;
    std::cout<<size_correspondence_result<<" Number of max correspondence found:"<<std::endl;
    float* correspondence_result = new float[size_correspondence_result];
    correspondenceRes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size_correspondence_result,correspondence_result,&ret);
    std::cout<<ret<<" Arg code 7.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,6,sizeof(correspondenceRes), &correspondenceRes);
    std::cout<<ret<<" Arg code 7.2 :"<<std::endl;

     //8. Arg correspondence_result_count;
    cl_mem corr_result = NULL;
    cl_int* corr_result_count = new cl_int[prod]();
    corr_result= clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*prod,corr_result_count,&ret);
    std::cout<<ret<< " Arg code 7.1:"<<std::endl;
    ret=clSetKernelArg(kernel,7,sizeof(corr_result),&corr_result);
    std::cout<<ret<< " Arg code 7.2:"<<std::endl;


    //9. Work size dimension
    cl_mem workSizeMemObj = NULL;
    int* worksizes = new int[2]();
    worksizes[0]= num_angle_steps;
    worksizes[1]= num_shift_steps;
    workSizeMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*2,worksizes,&ret);
    std::cout<<ret<< " Arg code 9.1 :"<<std::endl;
    ret=clSetKernelArg(kernel,8,sizeof(workSizeMemObj),&workSizeMemObj);
    std::cout<<ret<< " Arg code 9.2 :"<<std::endl;

    //10 and 11. To be shifted
    cl_mem sourceSizesMemObj = NULL;
    int* sources_sizes= new int[2]();
    sources_sizes[0]= static_cast<int>(model_voxelized->size());
    sources_sizes[1]= static_cast<int>(point_cloud_ptr->size());
    sourceSizesMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*2, sources_sizes, &ret);
    std::cout<<ret<< " Arg code 10.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,9,sizeof(sourceSizesMemObj),&sourceSizesMemObj);
    std::cout<<ret<< " Arg code 10.2  :"<<std::endl;

    //12. input_transformed
    cl_mem inputTransformedMemObj =NULL;
    int size_input_transformed_array =model_voxelized->size()*3*num_angle_steps*num_shift_steps;
    float* input_transformed_as_array = new float[size_input_transformed_array]();
    inputTransformedMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,sizeof(float)*size_input_transformed_array,input_transformed_as_array,&ret);
    std::cout<<ret<< " Arg code 11.1 :"<<std::endl;
    ret= clSetKernelArg(kernel,10,sizeof(inputTransformedMemObj),&inputTransformedMemObj);
    std::cout<<ret<< " Arg code 12.2:"<<std::endl;
     end = clock() ;
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for preparing kernel is : " <<elapsed_secs<<std::endl;


    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 2 , NULL,work_units, NULL, 0, NULL, NULL);
    std::cout<<"Running Program, code :" << ret <<std::endl;

    clFlush(command_queue);
    clFinish(command_queue);
    ret =  clEnqueueReadBuffer(command_queue,correspondenceRes,CL_TRUE,0,sizeof(float)*size_correspondence_result, &correspondence_result[0],0,NULL,NULL);
    std::cout<<"Reading Buffer, code : "<<ret<< std::endl;

    ret = clEnqueueReadBuffer(command_queue,corr_result,CL_TRUE,0,sizeof(int)*prod, &corr_result_count[0],0,NULL,NULL);
    std::cout<<"Reading Buffer, code :" << ret << std::endl;


    std::cout <<"Number of correspondence found of an instance is: " << corr_result_count[1] << std::endl;
    for ( int i = 0 ; i <20; i++) {
       // std::cout << correspondence_result[i]<< "   ";
         std::cout << corr_result_count[i]<< "  ";
    }

    std::cout<<std::endl;
      //free memory

    kernel = clCreateKernel(program,"computeDifferencesForCorrespondence", &ret);
    std::cout<<ret<<" Part 2 : "<<std::endl;

    cl_mem args_size_mem_obj = NULL;
    int *correspondence_count_real_size = new int( point_cloud_ptr->size());

    //TODO : WHY 4 times of this ?
    //TODO : Size of angle_count, shift_count is right the size of correspondent_count -> Must find size of correspondence_count
    float *angle_count = new float[point_cloud_ptr->size()*2];
    float *shift_count = new float[point_cloud_ptr->size()*2];

    args_size_mem_obj = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(int),correspondence_count_real_size,&ret);
    std::cout<<ret<<" Part 2.1.1 : "<<std::endl;
    ret = clSetKernelArg(kernel,1,sizeof(args_size_mem_obj), &args_size_mem_obj);
    std::cout<<ret<<" Part 2.1.2: "<<std::endl;

    pointCloudPtrMemObj = NULL ;
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(point_cloud_ptr_as_array),point_cloud_ptr_as_array,&ret);
    std::cout<<ret<<" Part 2.2.1: "<<std::endl;
    ret = clSetKernelArg(kernel,0,sizeof(pointCloudPtrMemObj), &pointCloudPtrMemObj);
    std::cout<<ret<<" Part 2.2.2 : "<<std::endl;


    cl_mem shift_count_mem_obj = NULL;
    shift_count_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(shift_count),shift_count,&ret);
    std::cout<<ret<<" Part 2.3.1 :"<<std::endl;
    ret = clSetKernelArg(kernel,3,sizeof(shift_count_mem_obj), &shift_count_mem_obj);
    std::cout<<ret<<" Part 2.3.2 : "<<std::endl;


    cl_mem angle_count_mem_obj = NULL;
    angle_count_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(angle_count),angle_count,&ret);
    std::cout<<ret<<" Part 2.4.1  : "<<std::endl;
    ret = clSetKernelArg(kernel,0,sizeof(angle_count_mem_obj), &angle_count_mem_obj);
    std::cout<<ret<<" Part 2.4.2  : "<<std::endl;



    //clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,work_units, NULL, 0, NULL, NULL);
    end = clock() ;
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time passed : " <<elapsed_secs<<std::endl;

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


    //https://streamhpc.com/blog/2013-04-28/opencl-error-codes/

    //https://stackoverflow.com/questions/26804153/opencl-work-group-concept

    //http://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html

    shift_and_roll_without_sum_in_cl(angleStart,angleEnd, angleStep,shiftStart, shiftEnd, shiftStep,
                                     correspondence_count, rotation,
                                     initialTranslation, std::get<1>(direction), model_voxelized,
                                     point_cloud_ptr);



    return 0;
}
