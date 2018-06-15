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
cl_mem workSizeMemObj,argsMemObj,modelVoxelizedMembObj, pointCloudPtrMemObj,inputTransformedMemObj,correspondenceRes,correspondenceResultCountMem =NULL;



cl_program  program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;


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

void findNextIteration(int *res, int numOfIteration, float *floatArgs) {
    //TODO : HERE JUST THE INDEX IS FOUND; HENCE ITS WRONG

    //find Max Angle,
    int max_angle_index = 0;
    float temp =0 ;
    for (int i = 0; i<numOfIteration; i++) {
        if (res[3*i+2]>temp) {
            temp=res[3*i+2];
            max_angle_index = res[3*i];
        }
    }

    //find Max Shift
    int max_shift_index = 0;
    temp = 0;
    for (int i = 0; i<numOfIteration; i++) {
        if (res[3*i+2]>temp) {
            temp=res[3*i+2];
            max_shift_index = res[3*i+1];
        }
    }

    //find Max Pair
    int max_pair_angle_index =0;
    int max_pair_shift_index =0;
    temp = 0;
    for (int i = 0; i<numOfIteration; i++) {
        if (res[3*i+2]>temp) {
            temp=res[3*i+2];
            max_pair_shift_index = res[3*i+1];
            max_pair_angle_index = res[3*i];
        }
    }

    float max_angle = max_angle_index*floatArgs[2]+floatArgs[0];
    float max_shift = max_shift_index*floatArgs[5]+floatArgs[3];

    //check min bound for values angle
    float angleStartNew = checkMinBoundsForValue(max_angle,floatArgs[0],floatArgs[2]);
    floatArgs[0]=angleStartNew;
    float angleEndNew = checkMaxBoundsForValue(max_angle,floatArgs[1],floatArgs[2]);
    floatArgs[1]=angleEndNew;

    float shiftStartNew = checkMinBoundsForValue(max_shift,floatArgs[3],floatArgs[5]);
    floatArgs[3]=shiftStartNew;
    float shiftEndNew = checkMaxBoundsForValue(max_shift,floatArgs[4],floatArgs[5]);
    floatArgs[4]=shiftEndNew;

    floatArgs[2] /= 5.0f;
    floatArgs[5] /= 5.0f;

    std::cout << "angle: " << max_angle << " Angle Start "<<angleStartNew<<" angle End " << angleEndNew<< " AngleStep " << floatArgs[2] <<std::endl;
    std::cout << "shift: " << max_shift <<" Shift Start "<<shiftStartNew <<" Shift End " << shiftEndNew<< "  Shift Step " << floatArgs[5] <<std::endl;



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

void prepareOpenCLProgramm(string kernel) {
    FILE *fp;
    //char fileName[] = "/home/tuan/Desktop/OpenCLBA-Local/OpenCLBA/kernel-original.cl";

    char *source_str;
    size_t source_size;

    fp = fopen(kernel.c_str(), "r");
    if (!fp) {
    fprintf(stderr, "Failed to load kernel\n");
    exit(1);
    }

    source_str = (char*)malloc(0x100000);
    source_size = fread(source_str,1,0x100000, fp);
    fclose(fp);

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    cl_queue_properties props[] = {
      CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
      0
    };


    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    program = clCreateProgramWithSource(context,1,(const char**)&source_str, (const size_t*)&source_size, &ret);

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
}

void shift_and_roll_without_sum_in_cl(float angle_min, float angle_max, float angle_step,
                                      float shift_min, float shift_max, float shift_step,
                                      std::vector<std::tuple<float, float, float>>& count,
                                      Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr

                                      ) {
    //DIAGRAMM Maker : https://live.amcharts.com/new/edit/
    //Initialize all variables

    clock_t end = clock();

    float args[21] ={angle_min, angle_max, angle_step, shift_min, shift_max, shift_step,initialTranslation[0],initialTranslation[1],initialTranslation[2],direction[0],direction[1],direction[2],rotation(0,0),rotation(0,1),rotation(0,2),rotation(1,0),rotation(1,1),rotation(1,2),rotation(2,0),rotation(2,1),rotation(2,2)};

    int* worksizes = new int[6]();
    worksizes[2]= model_voxelized->size();
    worksizes[5]= point_cloud_ptr->size();
    worksizes[4]= point_cloud_ptr.get()->size();

    int model_voxelized_as_array_size = (model_voxelized.get()->size())*3;
    float *model_voxelized_as_array = new float[model_voxelized_as_array_size]();
    convertPointCloudToCL(model_voxelized,model_voxelized_as_array,model_voxelized_as_array_size/3);
    modelVoxelizedMembObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , sizeof(float)*model_voxelized_as_array_size,model_voxelized_as_array,&ret);

    int point_cloud_ptr_array_size = (point_cloud_ptr.get()->size())*3;
    float* point_cloud_ptr_as_array = new float[point_cloud_ptr_array_size]();
    convertPointCloudToCL(point_cloud_ptr,point_cloud_ptr_as_array,point_cloud_ptr_array_size/3);
    pointCloudPtrMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*point_cloud_ptr_array_size, point_cloud_ptr_as_array,&ret);


    int num_angle_steps= std::round((args[1] - args[0]) / args[2]) + 1;
    int num_shift_steps = std::round((args[4] - args[3]) / args[5]) + 1;
    int old_prod = num_angle_steps*num_shift_steps;

    int size_input_transformed_array =worksizes[2]*3*num_angle_steps*num_shift_steps;
    int correspondenceResultCountSize =3*num_angle_steps*num_shift_steps;

    inputTransformedMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE |CL_MEM_ALLOC_HOST_PTR ,sizeof(float)*size_input_transformed_array,NULL,&ret);
    correspondenceResultCountMem = clCreateBuffer(context,CL_MEM_READ_WRITE |CL_MEM_ALLOC_HOST_PTR ,sizeof(int)*correspondenceResultCountSize,NULL,&ret);
    correspondenceRes = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR , sizeof(float)*size_input_transformed_array,NULL,&ret);

    for (int i = 0 ; i<4 ; i++) {
        //clock_t end = clock();

        int num_angle_steps= std::round((args[1] - args[0]) / args[2]) + 1;
        int num_shift_steps = std::round((args[4] - args[3]) / args[5]) + 1;
        int prod = num_angle_steps*num_shift_steps;
        std::cout<<"Number of steps taken "<<num_angle_steps<< " "<< num_shift_steps<<std::endl;

        worksizes[0]= num_angle_steps;
        worksizes[1]= num_shift_steps;
        worksizes[3]= model_voxelized->size()*num_angle_steps*num_shift_steps;
        workSizeMemObj = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, sizeof(int)*6,worksizes,&ret);

        int size_input_transformed_array =worksizes[2]*3*num_angle_steps*num_shift_steps;
        argsMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR ,21*sizeof(float),args,&ret);

        int correspondenceResultCountSize =3*num_angle_steps*num_shift_steps;
        int* correspondenceResultCount = new int[correspondenceResultCountSize]();

        size_t work_units[3] ={(size_t)num_angle_steps,(size_t)num_shift_steps, model_voxelized.get()->size()};
        size_t work_units2[1]= {(size_t)determinNumWorkItems(worksizes[2]*num_angle_steps*num_shift_steps)};

        /*
         *PART 1 : Transforming models
         */
        if (prod>old_prod)  {
            std::cout<<"Change Buffer"<<std::endl;
            inputTransformedMemObj = clCreateBuffer(context,CL_MEM_READ_WRITE |CL_MEM_ALLOC_HOST_PTR ,sizeof(float)*size_input_transformed_array,NULL,&ret);
            correspondenceResultCountMem = clCreateBuffer(context,CL_MEM_READ_WRITE |CL_MEM_ALLOC_HOST_PTR ,sizeof(int)*correspondenceResultCountSize,NULL,&ret);
            correspondenceRes = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR , sizeof(float)*size_input_transformed_array,NULL,&ret);
            old_prod=prod;
        }

        kernel = clCreateKernel(program,"transforming_models", &ret);

        ret = clSetKernelArg(kernel,0, sizeof(argsMemObj),&argsMemObj);
        ret = clSetKernelArg(kernel,1,sizeof(modelVoxelizedMembObj), &modelVoxelizedMembObj);
        ret = clSetKernelArg(kernel,2,sizeof(workSizeMemObj),&workSizeMemObj);
        ret= clSetKernelArg(kernel,3,sizeof(inputTransformedMemObj),&inputTransformedMemObj);

        std::cout<<"Read Buffer part 1, code:" << ret <<" Model size "<<model_voxelized.get()->size()<<"Point cloud "<<point_cloud_ptr.get()->size()<<std::endl;

        ret =  clEnqueueNDRangeKernel(command_queue, kernel, 3 , NULL,work_units, NULL, 0, NULL, NULL);
        //ret = clEnqueueReadBuffer(command_queue,inputTransformedMemObj,CL_TRUE,0,sizeof(float)*size_input_transformed_array, &input_transformed_as_array[0],0,NULL,NULL);

        //std::cout<<"Running Program part 1, code:" << ret <<std::endl;

        clFlush(command_queue);
        clFinish(command_queue);
        //clock_t end1 = clock();
        //double elapsed_secs = double(end1 - end) / CLOCKS_PER_SEC;
        //std::cout<<std::endl<<"Time needed for Step 1  : " <<elapsed_secs<<std::endl;

        /*
         *PART 2 : FIND CORRESPONDECES
         */

        kernel = clCreateKernel(program,"find_correspondences", &ret);

        ret = clSetKernelArg(kernel,0,sizeof(workSizeMemObj),&workSizeMemObj);
        ret = clSetKernelArg(kernel,1,sizeof(pointCloudPtrMemObj),&pointCloudPtrMemObj);
        ret = clSetKernelArg(kernel,2,sizeof(correspondenceRes), &correspondenceRes);
        ret= clSetKernelArg(kernel,3,sizeof(inputTransformedMemObj),&inputTransformedMemObj);

        size_t local_work_size[1] = {(size_t) 64};

        ret =  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,work_units2, local_work_size, 0, NULL, NULL);
        //std::cout<<"Running Program part 2, code:" << ret <<std::endl;

        clFlush(command_queue);
        clFinish(command_queue);
        //clock_t end2 = clock();
        //elapsed_secs = double(end2 - end1) / CLOCKS_PER_SEC;
        //std::cout<<std::endl<<"Time needed for Step 2  : " <<elapsed_secs<<std::endl;

        /*
         *PART 3 : Sum Up Result
         */
        kernel = clCreateKernel(program,"computeDifferencesForCorrespondence", &ret);
        ret = clSetKernelArg(kernel,0,sizeof(correspondenceRes),&correspondenceRes);
        ret = clSetKernelArg(kernel,1, sizeof(workSizeMemObj),&workSizeMemObj);
        ret = clSetKernelArg(kernel,2,sizeof(correspondenceResultCountMem),&correspondenceResultCountMem);

        size_t work_units3[2] ={(size_t)num_angle_steps,(size_t)num_shift_steps};

        ret =  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,work_units3,NULL, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue,correspondenceResultCountMem,CL_TRUE,0,sizeof(int)*correspondenceResultCountSize, &correspondenceResultCount[0],0,NULL,NULL);

        /*
        for (int m = 0 ; m < prod ; m++) {
            std::cout << correspondenceResultCount[3*m] << " "<< correspondenceResultCount[3*m+1] << " "<< correspondenceResultCount[3*m+2] << " "<< std::endl;
        }
        */

        clFlush(command_queue);
        clFinish(command_queue);
        //clock_t end3 = clock();
        //elapsed_secs = double(end3 - end2) / CLOCKS_PER_SEC;

        findNextIteration(correspondenceResultCount,prod,args);
        //std::cout<<std::endl<<"Time needed for Step 3  : " <<elapsed_secs<<std::endl;

        std::cout<<"Running Program part 3, code:" << ret <<std::endl<<std::endl<<std::endl;


    }
    clock_t end3 = clock();
    double elapsed_secs = double(end3 - end) / CLOCKS_PER_SEC;
    std::cout<<std::endl<<"Time needed for Step 3  : " <<elapsed_secs<<std::endl;

    clReleaseMemObject(modelVoxelizedMembObj);
    clReleaseMemObject(pointCloudPtrMemObj);
    clReleaseMemObject(inputTransformedMemObj);
    clReleaseMemObject(correspondenceResultCountMem);
    clReleaseMemObject(correspondenceRes);

    clReleaseMemObject(workSizeMemObj);
    clReleaseMemObject(argsMemObj);
}
void printHelp()
{
    pcl::console::print_error("Syntax is: .\oct_shift -models_dir -oct_dir -only_tip -shift -video <-video_dir>\n");
    pcl::console::print_info("  where arguments are:\n");
    pcl::console::print_info("                     -models_dir = directory where CAD model in .ply format is located, \n");
    pcl::console::print_info("                     -oct_dir = directory where OCT images are located, \n");
    pcl::console::print_info("                     -only_tip = 0 if whole OCT cloud should be used, 1 if only tip should be used, \n");
    pcl::console::print_info("                     -video = 1 if screenshots of algorithm for video should be taken, 0 if not, \n");
    pcl::console::print_info("                     -video_dir = necessary if video is set to 1. \n");
}


int main(int argc, char **argv)
{
    if (argc<2) {
        printHelp();
        return -1;
    }

    string path = argv[1];
    string oct_dir = argv[2];
    string kernel_path = argv[3];

    //pcl::console::parse_argument(argc, argv, "-models_dir", path);
    //pcl::console::parse_argument(argc, argv, "-oct_dir", oct_dir);

    //TODO : Parameterize this please
    //std::string path= "/home/tuan/Desktop/Models";
    //std::string oct_dir ="/home/tuan/Desktop/042801/";

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

    prepareOpenCLProgramm(kernel_path);
    shift_and_roll_without_sum_in_cl(angleStart,angleEnd, angleStep,shiftStart, shiftEnd, shiftStep, correspondence_count, rotation,initialTranslation, std::get<1>(direction), model_voxelized, point_cloud_ptr);

    //shift_and_roll_without_sum_in_cl(-3.5,0.5, 0.2,0.3,0.5, 0.01, correspondence_count, rotation,initialTranslation, std::get<1>(direction), model_voxelized, point_cloud_ptr);
    //TEST 1 : Anglemin = angleStart

    return 0;
}
