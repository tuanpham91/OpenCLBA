#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//https://stackoverflow.com/questions/36410745/how-to-pass-c-vector-of-vectors-to-opencl-kernel


void rigidTransformationCL (int size,float *input, float4 transformation_matrix, float *input_transformed) {
  for (int i = 0; i< size ; i++) {
      float temp = input[4*i];
      input_transformed[4*i] = temp*transformation_matrix[0] + input[4*i+1]*transformation_matrix[1] + input[4*i+2]*transformation_matrix[2]+ input[4*i+3]*transformation_matrix[3];
      input_transformed[4*i+1] = temp*transformation_matrix[4] + input[4*i+1]*transformation_matrix[5] + input[4*i+2]*transformation_matrix[6]+ input[4*i+3]*transformation_matrix[7];
      input_transformed[4*i+2] = temp*transformation_matrix[8] + input[4*i+1]*transformation_matrix[9] + input[4*i+2]*transformation_matrix[10]+ input[4*i+3]*transformation_matrix[11];
      input_transformed[4*i+3] = temp*transformation_matrix[12] + input[4*i+1]*transformation_matrix[13] + input[4*i+2]*transformation_matrix[14]+ input[4*i+3]*transformation_matrix[15];
  }
}

float checkMinBoundsForValueCL(float value, float start, float step) {
	float val = value - step;
	if (val > start) {
		if (val - step >= start) {
			return val - step;
		}
		return val;
	}
	return start;
}

float checkMaxBoundsForValueCL(float value, float end, float step) {
	float val = value + step;
	if (val < end) {
		if (val + step <= end) {
			return val + step;
		}
		return val;
	}
	return end;
}


int findMaxIndexOfVectorOfTuplesCL(__global float *tuples,__global int *size) {
  int max_index =0;
  float max= 0.0f;
  //TODO : Review this
  for (int i = 0 ; i < size[0] ; i++) {
    if (tuples[i*3+2]>max) {
      max = tuples[i*3+2];
      max_index = i;
    }
  }
  return max_index;
}

int findMaxIndexOfVectorOfPairsCL(__global float *angle_count,__global int *size) {
  int max_index =0;
  float max= 0.0f;
  for (int i = 0 ; i < size[0] ; i++) {
    if (angle_count[3*i+1]>max) {
      max = angle_count[i*3+1];
      max_index = i;
    }
  }
  return max_index;
}


__kernel void find_correspondences(__global int *intArgs, __global float *point_cloud_ptr, __global float *correspondence_result, __global int *sources_size, __global float *input_transformed) {

  __private int i = get_global_id(0);
  __private int max_number_of_points = intArgs[1];

  if (i >= max_number_of_points) {
    return;
  }

  __private int point_cloud_ptr_size = sources_size[1];

  float a = 0.0;
  float b = 0.0;
  float c = 0.0;

  for (int k = 0; k< point_cloud_ptr_size; k++  ) {
    a = (input_transformed[3*i] - point_cloud_ptr[3*k])*(input_transformed[3*i] - point_cloud_ptr[3*k]);
    b = (input_transformed[3*i+1] - point_cloud_ptr[3*k+1])*(input_transformed[3*i+1] - point_cloud_ptr[3*k+1]);
    c = (input_transformed[3*i+2] - point_cloud_ptr[3*k+2])*(input_transformed[3*i+2] - point_cloud_ptr[3*k+2]);
    //if (dis<=0.5) {
    if (a+b+c<1.5) {
      correspondence_result[3*i]= (float)i;
      correspondence_result[3*i+1] =(float)k;
      correspondence_result[3*i+2] = a+b+c;
      k=point_cloud_ptr_size;
    }
  }
  //Subject to Change
  //correspondence_result_count[i]=a+b+c;
}


// TUAN : Line 374 global_classification
/*
  1. floatArgs : Collections of arguments with float data type like : float angle_min, float angle_max, float angle_step, float shift_min, float shift_max, float shift_step,
  2. initialTranslation :
  3. direction:
  4. model_voxelized: to be shifted PointCloud (correspondence)
  5. point_cloud_ptr: original PointCloud (correspondence)
  6. rotation
  7. correspondence_result: result of all between point clouds
  8. correspondence_count : count of correspondences
  9. work_size_dimension :
  10. model_voxelized_size:
  11. point_cloud_ptr_size:
*/
__kernel void shiftAndRollWithoutSumLoop(__global float *floatArgs, __global float *initialTranslation, __global float *direction,__global float *model_voxelized, __global float *point_cloud_ptr, __global float *rotation, __global float *correspondence_result,__global int *correspondence_result_count, __global int *work_size_dimension, __global int *sources_size, __global float *input_transformed) {
}

/*
  floatArgs:
  initialTranslation : 6-8
  direction :  9-11
  rotation :12-20 (TODO: WHY IS IT NOT USED)
*/
__kernel void transforming_models(__global float *floatArgs,__global float *model_voxelized, __global int *work_size_dimension,  __global float *input_transformed) {

  __private int angle = get_global_id(0);
  __private int shift = get_global_id(1);
  __private int point = get_global_id(2);

  __private float angle_min = floatArgs[0];
  __private float angle_max = floatArgs[1];

  __private float angle_step = floatArgs[2];
  __private float shift_min  = floatArgs[3];

  __private float shift_max = floatArgs[4];
  __private float shift_step = floatArgs[5];


  __private int number_angle_step = work_size_dimension[0];
  __private int number_shift_step = work_size_dimension[1];
  __private int model_voxelized_size = work_size_dimension[2];

  //Space holder for shifted point Cloud
  __private float rot[9] = {};
  __private float trans[3]= {};
  __private float transform[16]= {};

  __private int start_index = (number_shift_step*angle+shift)*model_voxelized_size;

  //This methode is replaced by following lines :
  //rotateByAngleCL(angle_min+ angle*angle_step, rot);

  __private float angle_temp =(angle_min+angle*angle_step)*(0.01745328888);
  rot[0] = cos(angle_temp);
  rot[1] = -sin(angle_temp);
  rot[2] = 0.0f;
  rot[3] = sin(angle_temp);
  rot[4] = cos(angle_temp);
  rot[5] = 0.0f;
  rot[6] = 0.0f;
  rot[7] = 0.0f;
  rot[8] = 1.0f;

  //This methode is replaced by following lines :
  //shiftByValueCL(shift_min+ shift*shift_step, initialTranslation, direction, trans);
  //TODO 3/5 : maybe something wrong here
  __private float shift_temp = shift_min + shift*shift_step;
  trans[0] = floatArgs[6]*shift_temp/floatArgs[11];
  trans[1] = floatArgs[7]*shift_temp/floatArgs[11];
  trans[2] = floatArgs[8]*shift_temp/floatArgs[11];

  //This methode is replaced by following lines :
  //buildTransformationMatrixCL(rot,trans,transform);
  transform[0] = rot[0];
  transform[1] = rot[1];
  transform[2] = rot[2];
  transform[3] = rot[3];
  transform[4] = rot[4];
  transform[5] = rot[5];
  transform[6] = rot[6];
  transform[7] = rot[7];
  transform[8] = rot[8];
  transform[9] = trans[0];
  transform[10] = trans[1];
  transform[11] = trans[2];
  transform[12] = 0;
  transform[13] = 0;
  transform[14] = 0;
  transform[15] = 1;

  __private float max_distance_sqr = (float) 0.0004f;

  //computeCorrespondencesCL(transform,model_voxelized,point_cloud_ptr, correspondence_result, model_voxelized_size, point_cloud_ptr_size,input_transformed);

  __private bool ident = true;
  __private int i = 0;
  __private int k = 0;

  for (i = 0 ; i < 4 ; i++) {
    for (k = 0; k < 4 ; k++) {
      if (i == k ) {
        if (transform[i*4+k]!= 1.0f) {
          ident = false;
          break;
        }
      }
      else {
        if(transform[i*4+k]!= 0.0f) {
          ident = false;
          break;
        }
      }
    }
  }

  i = point;
  if (!ident) {
    input_transformed[start_index + 3*i] = model_voxelized[3*i]*transform[0] + model_voxelized[3*i+1]*transform[4] + model_voxelized[ 3*i+2]*transform[8]+transform[3];
    input_transformed[start_index + 3*i+1] = model_voxelized[3*i]*transform[1] + model_voxelized[3*i+1]*transform[5] + model_voxelized[3*i+2]*transform[9]+transform[7];
    input_transformed[start_index + 3*i+2] = model_voxelized[3*i]*transform[2] + model_voxelized[3*i+1]*transform[6] + model_voxelized[3*i+2]*transform[10]+transform[11];

  }
  else {
    input_transformed[start_index+3*i]=model_voxelized[3*i];
    input_transformed[start_index+3*i+1]=model_voxelized[3*i+1];
    input_transformed[start_index+3*i+2]=model_voxelized[3*i+2];
  }
}

/* TODO : Evaluate this :
    Schema
    OLD ----- NEW
    correspondence_count -> correspondence_result
    size_correspondence_count -> work_sizes[2]
    angle_count, shift_count -> WTF
*/
__kernel void computeDifferencesForCorrespondence(__global float *correspondence_result,__global float *floatArgs,  __global int *work_sizes,  __global float *angle_count, __global float *shift_count) {
    int i  = get_global_id(0);

    float angle_temp = correspondence_result[i];
    float shift_temp = correspondence_result[i+1];
    float count_temp = correspondence_result[i+2];

    __private float angleStart = floatArgs[0];
    __private float angleEnd = floatArgs[1];

    __private float angleStep = floatArgs[2];
    __private float shiftStart  = floatArgs[3];

    __private float shiftEnd = floatArgs[4];
    __private float shiftStep = floatArgs[5];

    __private int size_correspondence_count = work_sizes[2];
    //4.5 : TODO : from here

    float **iterator;
    int iter_help;
    for (int  i = 0; i < size_correspondence_count ; i++) {
      if (angle_count[i]==angle_temp) {
        iter_help = i;
        break;
      }
    }

    if (iter_help != size_correspondence_count) {
      angle_count[iter_help+1] += count_temp;
    } else {
      //Add more into angle_count
      //TODO : Size angle count here is not defined
      //angle_count. //TODO :
    }

    for (int  i = 0; i < size_correspondence_count ; i++) {
      if (shift_count[i]==shift_temp) {
        iter_help = i;
        break;
      }
    }

    if (iter_help != size_correspondence_count) {
      shift_count[iter_help+1] += shift_temp;
    } else {
      //angle_count.push_back //TODO :
    }
    //TODO : Which size here
    /*int max_index_angles= findMaxIndexOfVectorOfPairsCL(angle_count,size_correspondence_count);
    //TODO : Which size here
    int max_index_shift = findMaxIndexOfVectorOfPairsCL(shift_count,size_correspondence_count);

    int correspondence_index = findMaxIndexOfVectorOfTuplesCL(correspondence_count, size_correspondence_count);

    ;
    float max_angle = angle_count[max_index_angles];
    float max_shift = angle_count[max_index_angles];

    angleStart = checkMinBoundsForValueCL(max_angle,angleStart,angleStep);
    angleEnd = checkMaxBoundsForValueCL(max_angle,angleEnd, angleStep);
    shiftStart = checkMinBoundsForValueCL(max_shift,shiftStart,shiftStep);
    shiftEnd = checkMaxBoundsForValueCL(max_shift,shiftEnd,shiftStep);
    angleStep= angleStep/5.0f;
    shiftStep= shiftStep/5.0f;
    //TODO :
    */
}

//https://stackoverflow.com/questions/7627098/what-is-a-lambda-expression-in-c11
/*
  Files to do this :
  1. https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/impl/correspondence_estimation.hpp#L113
  2. https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/correspondence_estimation.h#L63
  3. https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/pcl_base.h
*/
