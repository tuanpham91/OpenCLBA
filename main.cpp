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
#include "misc.h"
#include "util.h"
#include "regression.h"
#include "oct_processing.h"
#include "graphUtils/GraphUtils.h"
#include "transformations.h"


#include <CL/cl2.hpp>
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


void testCreatingMatrix(float *floatArgs) {

     float angle_min = floatArgs[0];
     float angle_max = floatArgs[1];

     float angle_step = floatArgs[2];
     float shift_min  = floatArgs[3];

     float shift_max = floatArgs[4];
     float shift_step = floatArgs[5];



      //Space holder for shifted point Cloud
      float rot[9] = {};
      float source[9]= {};
      float rotating[9]= {};


      for (int m = 0 ; m<11 ; m++) {
          for (int n = 0; n<11; n++) {
              float transform[16]= {};

              float angle_temp =(angle_min+m*angle_step)*(0.01745328888);
              rotating[0] = cos(angle_temp);
              rotating[1] = -sin(angle_temp);
              rotating[2] = 0.0f;
              rotating[3] = sin(angle_temp);
              rotating[4] = cos(angle_temp);
              rotating[5] = 0.0f;
              rotating[6] = 0.0f;
              rotating[7] = 0.0f;
              rotating[8] = 1.0f;

              transform[0]= floatArgs[12]*rotating[0]+floatArgs[13]*rotating[3]+floatArgs[14]*rotating[6];
              transform[1]= floatArgs[12]*rotating[1]+floatArgs[13]*rotating[4]+floatArgs[14]*rotating[7];
              transform[2]= floatArgs[12]*rotating[2]+floatArgs[13]*rotating[5]+floatArgs[14]*rotating[8];

              transform[4]= floatArgs[15]*rotating[0]+floatArgs[16]*rotating[3]+floatArgs[17]*rotating[6];
              transform[5]= floatArgs[15]*rotating[1]+floatArgs[16]*rotating[4]+floatArgs[17]*rotating[7];
              transform[6]= floatArgs[15]*rotating[2]+floatArgs[16]*rotating[5]+floatArgs[17]*rotating[8];

              transform[8]= floatArgs[18]*rotating[0]+floatArgs[19]*rotating[3]+floatArgs[20]*rotating[6];
              transform[9]= floatArgs[18]*rotating[1]+floatArgs[19]*rotating[4]+floatArgs[20]*rotating[7];
              transform[10]= floatArgs[18]*rotating[2]+floatArgs[19]*rotating[5]+floatArgs[20]*rotating[8];


              float shift_temp = shift_min + n*shift_step;
              transform[3] = floatArgs[6]+ floatArgs[9]*shift_temp/floatArgs[11];
              transform[7] =floatArgs[7]+ floatArgs[10]*shift_temp/floatArgs[11];
              transform[11] =floatArgs[8]+ floatArgs[11]*shift_temp/floatArgs[11];

              transform[12] = 0;
              transform[13] = 0;
              transform[14] = 0;
              transform[15] = 1;

              for (int k =0; k<4 ; k++) {
                  std::cout<<transform[k*4]<<" "<<transform[k*4+1]<<" "<<transform[k*4+2]<<" "<<transform[k*4+3]<<" "<<std::endl;
              }
          }
      }

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

void processOTCFrame(cv::Mat imageGray,int number , boost::shared_ptr<std::vector<std::tuple<int,int, cv::Mat, cv::Mat>>> needle_width  ) {

    cv::Mat transposedOCTimage;
        cv::flip(imageGray, imageGray, 0);

        //set a threshold (0.26)
        cv::Mat thresholdedImage;
        cv::threshold(imageGray, thresholdedImage, 0.26 * 255, 1, 0);

        //use a median blur filter
        cv::Mat filteredImage;
        cv::medianBlur(thresholdedImage, filteredImage, 3);

        //label the image
        cv::Mat labelledImage;
        cv::Mat labelStats;
        cv::Mat labelCentroids;
        int numLabels = cv::connectedComponentsWithStats(filteredImage, labelledImage, labelStats, labelCentroids);

        //for every label with more than 400 points process it further for adding points to the cloud
        for (int i = 1; i < numLabels; i++) {
            //original threshold at 400
            if (labelStats.at<int>(i, cv::CC_STAT_AREA) > 250) {
                cv::Mat labelInfo = labelStats.row(i);
                //save bounding box width for finding the point where needle gets smaller
                needle_width->push_back(std::tuple<int, int, cv::Mat, cv::Mat>(number, labelStats.at<int>(i, cv::CC_STAT_WIDTH), filteredImage, labelInfo));
            }
        }
}

boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> recognizeOTC(pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points, std::string oct_dir, bool only_tip ) {
    std::string oct_directory = getDirectoryPath(oct_dir);
    //count oct images
    int fileCount = 128;
    //countNumberOfFilesInDirectory(oct_directory, "%s*.bmp");
    int minFrameNumber = 0;
    int maxFrameNumber = fileCount;

    //tuple with frame number, bounding box width, filteredImage, labelInfo
    boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width(new std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>);
    cv::Mat imageGray;
        {
            pcl::ScopeTime t("Process OCT images");
            //	go through all frames
            for (int number = minFrameNumber; number < maxFrameNumber; number++)
            {
                //get the next frame
                std::stringstream filename;
                if (number < 100) {
                    filename << "0";
                }
                if (number < 10) {
                    filename << "0";
                }
                filename << number << ".bmp";
                //read the image in grayscale
                imageGray = cv::imread(oct_dir + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

                processOCTFrame(imageGray, number, needle_width);

                cv::waitKey(10);
            }

            //---------------------------------------------
            //optionally cut needle tip off
            //---------------------------------------------
            int end_index = needle_width->size();
            //regression to find cutting point where tip ends
            if (only_tip) {
                end_index = regression(needle_width);
            }
            //go through all frames
            for (int w = 0; w < end_index; w++) {
                std::tuple<int, int, cv::Mat, cv::Mat> tup = needle_width->at(w);
                std::vector<cv::Point> elipsePoints;
                MatToPointXYZ(std::get<2>(tup), std::get<3>(tup), elipsePoints, std::get<0>(tup), point_cloud_ptr, imageGray.rows, imageGray.cols);

                //compute center point of needle frame for translation
                if (elipsePoints.size() >= 50) { //to remove outliers, NOT RANSAC
                    cv::RotatedRect elipse = cv::fitEllipse(cv::Mat(elipsePoints));
                    pcl::PointXYZ peak;
                    generatePoint(peak, elipse.center.x, elipse.center.y, std::get<0>(tup), imageGray.cols, imageGray.rows);
                    peak_points->push_back(peak);
                }
            }
        }

    //downsample pointcloud OCT
    float VOXEL_SIZE_ICP_ = 0.02f;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
    voxel_grid_icp.setInputCloud(point_cloud_ptr);
    voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
    voxel_grid_icp.filter(*point_cloud_ptr);

    return needle_width;
}

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

    std::cout << "Number of Point clouds :  " << point_cloud_ptr->size()<< " " <<model_voxelized->size()<< std::endl;

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

    /*
     *PART 1 : Transforming models
     */

    kernel = clCreateKernel(program,"transforming_models", &ret);

    float args[21] ={angle_min, angle_max, angle_step, shift_min, shift_max, shift_step,initialTranslation[0],initialTranslation[1],initialTranslation[2],direction[0],direction[1],direction[2],rotation(0,0),rotation(0,1),rotation(0,2),rotation(1,0),rotation(1,1),rotation(1,2),rotation(2,0),rotation(2,1),rotation(2,2)};
    for (int k = 0; k <21 ; k++) {
        std::cout<<args[k]<< " ";
    }
    argsMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR ,21*sizeof(float),args,&ret);
    ret = clSetKernelArg(kernel,0, sizeof(argsMemObj),(void *)&argsMemObj);

    //4. Arg model_voxelized
    int model_voxelized_as_array_size = static_cast<int>(model_voxelized.get()->size())*3;
    float *model_voxelized_as_array = new float[model_voxelized_as_array_size]();
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array,model_voxelized_as_array_size/3);
    modelVoxelizedMembObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , sizeof(float)*model_voxelized_as_array_size,model_voxelized_as_array,&ret);
    ret = clSetKernelArg(kernel,1,sizeof(modelVoxelizedMembObj), &modelVoxelizedMembObj);

    //9. Work size dimension
    cl_mem workSizeMemObj = NULL;
    int* worksizes = new int[3]();
    worksizes[0]= num_angle_steps;
    worksizes[1]= num_shift_steps;
    worksizes[2]= static_cast<int>(model_voxelized->size());
    std::cout<< "Number of steps "<< num_angle_steps<< " " << num_shift_steps<< std::endl;
    workSizeMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*2,worksizes,&ret);
    ret=clSetKernelArg(kernel,2,sizeof(workSizeMemObj),&workSizeMemObj);

    //12. input_transformed
    cl_mem inputTransformedMemObj =NULL;
    int size_input_transformed_array =worksizes[2]*3*num_angle_steps*num_shift_steps;
    float* input_transformed_as_array = new float[size_input_transformed_array]();
    inputTransformedMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,sizeof(float)*size_input_transformed_array,input_transformed_as_array,&ret);
    ret= clSetKernelArg(kernel,3,sizeof(inputTransformedMemObj),&inputTransformedMemObj);

    size_t work_units[3] ={(size_t)num_angle_steps_s,(size_t)num_shift_steps_s, model_voxelized.get()->size()};
    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 3 , NULL,work_units, NULL, 0, NULL, NULL);
    std::cout<<"Running Program, code:" << ret <<std::endl;

    clFlush(command_queue);
    clFinish(command_queue);
    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 3 , NULL,work_units, NULL, 0, NULL, NULL);
    std::cout<<"Running Program, code:" << ret <<std::endl;


    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for 1. kernel method is : " <<elapsed_secs<<std::endl;
    std::cout<<"DEBUG : Last elements of input transformed is " << input_transformed_as_array[2460774]<< " "<<input_transformed_as_array[2460775]<< " "<<input_transformed_as_array[2460776]<< std::endl;

    ret = clEnqueueReadBuffer(command_queue,inputTransformedMemObj,CL_TRUE,0,sizeof(float)*size_input_transformed_array, &input_transformed_as_array[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;
    /*
     *PART 2 : Transforming models
     */

    kernel = clCreateKernel(program,"find_correspondences", &ret);
    std::cout<<ret<<" Part 2 :find correspondence "<<std::endl;
    int number_of_points_to_calculate = worksizes[2]*num_angle_steps*num_shift_steps;
    int int_args[2] = {1,number_of_points_to_calculate};
    cl_mem intArgs = NULL;
    intArgs = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int)*2,int_args,&ret);
    ret = clSetKernelArg(kernel,0,sizeof(intArgs), &intArgs);

    size_t work_units2[1]= {(size_t)determinNumWorkItems(number_of_points_to_calculate)};
    int point_cloud_ptr_array_size = static_cast<int>(point_cloud_ptr.get()->size())*3;
    float* point_cloud_ptr_as_array = new float[point_cloud_ptr_array_size]();
    convertPointCloudToCL(point_cloud_ptr,point_cloud_ptr_as_array,point_cloud_ptr_array_size/3);
    std::cout<< "Size of pointCloud array is " << point_cloud_ptr_array_size<< " points , last value is: " << point_cloud_ptr_as_array[point_cloud_ptr_array_size-1]<< " compare with "<< point_cloud_ptr.get()->at(point_cloud_ptr_array_size/3-1).z <<std::endl;
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*point_cloud_ptr_array_size, point_cloud_ptr_as_array,&ret);
    ret = clSetKernelArg(kernel,1,sizeof(pointCloudPtrMemObj),&pointCloudPtrMemObj);

    cl_mem correspondenceRes= NULL;
    int size_correspondence_result = static_cast<int>(model_voxelized->size())*3*num_angle_steps*num_shift_steps;
    std::cout<<"DEBUG : Number of max correspondence found:"<<size_correspondence_result<<std::endl;
    float* correspondence_result = new float[size_correspondence_result]();
    correspondenceRes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float)*size_correspondence_result,correspondence_result,&ret);
    ret = clSetKernelArg(kernel,2,sizeof(correspondenceRes), &correspondenceRes);

    cl_mem sourceSizesMemObj = NULL;
    int *sources_sizes= new int[2]();
    sources_sizes[0]= static_cast<int>(model_voxelized->size());
    sources_sizes[1]= static_cast<int>(point_cloud_ptr_array_size/3);
    sourceSizesMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*2, sources_sizes, &ret);
    ret = clSetKernelArg(kernel,3,sizeof(sourceSizesMemObj),&sourceSizesMemObj);


    ret= clSetKernelArg(kernel,4,sizeof(inputTransformedMemObj),&inputTransformedMemObj);
    //std::cout<<ret<<" Part 2.1.7 : "<<std::endl;

    size_t local_work_size[1] = {(size_t) 64};
    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,work_units2, local_work_size, 0, NULL, NULL);
    std::cout<<"Running Program part 2, code:" << ret <<std::endl;

    clFlush(command_queue);
    clFinish(command_queue);
    /*
    ret = clEnqueueReadBuffer(command_queue,corr_result,CL_TRUE,0,sizeof(float)*corr_result_size, &corr_result_count[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;
    */
    /*
    ret = clEnqueueReadBuffer(command_queue,correspondenceRes,CL_TRUE,0,sizeof(float)*size_correspondence_result, &correspondence_result[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;
    */
    //for ( int i = corr_result_size-100; i <corr_result_size; i++) {

    /*for ( int i = 0; i <100; i++) {
       //std::cout << corr_result_count[i]<<"  ";
        //std::cout << corr_result_count[i]<<"  ";
        std::cout<<correspondence_result[i]<<"  ";
    }
    */

    clock_t end2 = clock() ;
    elapsed_secs = double(end2 - end) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for 2. kernel method is : " <<elapsed_secs<<std::endl;

    /*
     *PART 3 : Sum Up Result
     */


    kernel = clCreateKernel(program,"computeDifferencesForCorrespondence", &ret);
    std::cout<<ret<<" Part 3.: "<<std::endl;

    ret = clSetKernelArg(kernel,0,sizeof(correspondenceRes),&correspondenceRes);
    std::cout<<ret<<" Part 3.1: "<<std::endl;

    ret = clSetKernelArg(kernel,1, sizeof(argsMemObj),&argsMemObj);
    std::cout<<ret<<" Part 3.1: "<<std::endl;

    ret = clSetKernelArg(kernel,2, sizeof(workSizeMemObj),&workSizeMemObj);
    std::cout<<ret<<" Part 3.2: "<<std::endl;

    cl_mem correspondenceResultCountMem =NULL;
    int correspondenceResultCountSize =3*num_angle_steps*num_shift_steps;
    int* correspindenceResultCount = new int[correspondenceResultCountSize]();
    correspondenceResultCountMem = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,sizeof(int)*correspondenceResultCountSize,correspindenceResultCount,&ret);
    std::cout<<ret<< " Arg code 3.3 :"<<std::endl;
    ret= clSetKernelArg(kernel,3,sizeof(correspondenceResultCountMem),&correspondenceResultCountMem);
    std::cout<<ret<< " Arg code 3.3:"<<std::endl;
    clock_t end3 = clock() ;

    size_t work_units3[2] ={(size_t)num_angle_steps_s,(size_t)num_shift_steps_s};
    std::cout<<"Work units :" << work_units3[0] << " "<< work_units3[1]<<std::endl;

    ret =  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,work_units3,NULL, 0, NULL, NULL);
    std::cout<<"Running Program part 3, code:" << ret <<std::endl;

    ret = clEnqueueReadBuffer(command_queue,correspondenceResultCountMem,CL_TRUE,0,sizeof(int)*correspondenceResultCountSize, &correspindenceResultCount[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;

    ret = clEnqueueReadBuffer(command_queue,correspondenceRes,CL_TRUE,0,sizeof(float)*size_correspondence_result, &correspondence_result[0],0,NULL,NULL);
    std::cout<<"Reading Buffer , code :" << ret << std::endl;
    for ( int i = 0; i <100; i++) {
        std::cout<<correspindenceResultCount[i]<<"  ";
    }
    elapsed_secs = double(end3 - end) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for 3. kernel method is : " <<elapsed_secs<<std::endl;


    testCreatingMatrix(args);
    /*for (int i = 0 ;i< model_voxelized->size();i++) {
        std::cout <<input_transformed_as_array[3*i]<<"  "<<input_transformed_as_array[3*i+1]<<"  "<<input_transformed_as_array[3*i+2]<<"  "<<std::endl;
    }
    */
}


int main()
{
    //TODO : Parameterize this please
    std::string path= "/home/tuan/Desktop/Models";
    std::string oct_dir ="/home/tuan/Desktop/042801/";

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_not_cut(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr peak_points(new pcl::PointCloud<pcl::PointXYZ>);
    boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width = recognizeOTC(point_cloud_not_cut, peak_points, oct_dir, false);
    /*
    viewer.showCloud(point_cloud_not_cut);
    while (!viewer.wasStopped ())   {
    }
    */
    //Copied Code

    pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized(new pcl::PointCloud<pcl::PointXYZ>());

    generatePointCloudFromModel(modelCloud, model_voxelized, path);

   // std::cout<<"DEBUG : getModelSize : "<< getModelSize(model_voxelized)<< " Get MinZ Value : " << getMinZValue(point_cloud_not_cut)<<std::endl;
    cutPartOfModel(point_cloud_not_cut, point_cloud_ptr, getModelSize(model_voxelized) - 0.1f + getMinZValue(point_cloud_not_cut));

    std::pair<Eigen::Vector3f, Eigen::Vector3f> direction = computeNeedleDirection(peak_points);

    std::cout << "origin: " << std::endl << std::get<0>(direction) << std::endl << "direction: " << std::endl << std::get<1>(direction) << std::endl;

    Eigen::Matrix3f rotation = computeNeedleRotation(direction);
    std::cout << "rotation matrix: " << std::endl << rotation << std::endl;

    //TODO :CHeck this
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

    //TEST 1 : Anglemin = angleStart

    return 0;
}



/*
std::cout<<"IST"<<std::endl;

float rot[9] = {};
float source[9]={};
convertMatrix3fToCL(rotation,source);
float angle_temp =(angleStart)*(0.01745328888);
rot[0] = cos(angle_temp);
rot[1] = -sin(angle_temp);
rot[2] = 0.0f;
rot[3] = sin(angle_temp);
rot[4] = cos(angle_temp);
rot[5] = 0.0f;
rot[6] = 0.0f;
rot[7] = 0.0f;
rot[8] = 1.0f;

float res[9] = {};
res[0]= source[0]*rot[0]+source[1]*rot[3]+source[2]*rot[6];
res[1]= source[0]*rot[1]+source[1]*rot[4]+source[2]*rot[7];
res[2]= source[0]*rot[2]+source[1]*rot[5]+source[2]*rot[8];

res[3]= source[3]*rot[0]+source[4]*rot[3]+source[5]*rot[6];
res[4]= source[3]*rot[1]+source[4]*rot[4]+source[5]*rot[7];
res[5]= source[3]*rot[2]+source[4]*rot[5]+source[5]*rot[8];

res[6]= source[6]*rot[0]+source[7]*rot[3]+source[8]*rot[6];
res[7]= source[6]*rot[1]+source[7]*rot[4]+source[8]*rot[7];
res[8]= source[6]*rot[2]+source[7]*rot[5]+source[8]*rot[8];

for (int i = 0 ; i<3 ; i++) {
    std::cout<<res[i*3]<< "  "<<res[i*3+1]<<"  "<<res[i*3+2]<<std::endl;
}

Eigen::Matrix3f rotation1;

std::cout<<std::endl<<"SOLL"<<std::endl;

std::cout<<rotateByAngle(angleStart,rotation)<<std::endl;
*/
