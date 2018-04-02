#pragma OPENCL EXTENSION cl_khr_fp64 : enable


void rotateByAngleCL(float angleInDegrees, float *res) {
  float angle = (float)(angleInDegrees*(0.01745328888));
  res[0] = cos(angle);
  res[1] = -sin(angle);
  res[2] = 0.0f;
  res[3] = sin(angle);
  res[4] = cos(angle);
  res[5] = 0.0f;
  res[6] = 0.0f;
  res[7] = 0.0f;
  res[8] = 1.0f;
}

//  shift_and_roll_without_sum
void shiftByValueCL(float shift,__global float *currentTranslation,__global float *direction,float *trans) {
  trans[0] = currentTranslation[0]*shift/direction[2];
  trans[1] = currentTranslation[1]*shift/direction[2];
  trans[2] = currentTranslation[2]*shift/direction[2];
}

//https://stackoverflow.com/questions/36410745/how-to-pass-c-vector-of-vectors-to-opencl-kernel

//  shift_and_roll_without_sum
void buildTransformationMatrixCL(float *rotation, float *translation, float *transformation ) {
  transformation[0] = rotation[0];
  transformation[1] = rotation[1];
  transformation[2] = rotation[2];
  transformation[3] = rotation[3];
  transformation[4] = rotation[4];
  transformation[5] = rotation[5];
  transformation[6] = rotation[6];
  transformation[7] = rotation[7];
  transformation[8] = rotation[8];
  transformation[9] = translation[0];
  transformation[10] = translation[1];
  transformation[11] = translation[2];
  transformation[12] = 0;
  transformation[13] = 0;
  transformation[14] = 0;
  transformation[15] = 1;
}
float calculate_distance(float *pointA,float *pointB)  {
  float a = (pointA[0] - pointB[0])*(pointA[0] - pointB[0]);
  float b = (pointA[1] - pointB[1])*(pointA[1] - pointB[1]);
  float c = (pointA[2] - pointB[2])*(pointA[2] - pointB[2]);
  return sqrt(a+b+c);
}
void rigidTransformationCL (int size,float *input, float4 transformation_matrix, float *input_transformed) {
  for (int i = 0; i< size ; i++) {
      float temp = input[4*i];
      input_transformed[4*i] = temp*transformation_matrix[0] + input[4*i+1]*transformation_matrix[1] + input[4*i+2]*transformation_matrix[2]+ input[4*i+3]*transformation_matrix[3];
      input_transformed[4*i+1] = temp*transformation_matrix[4] + input[4*i+1]*transformation_matrix[5] + input[4*i+2]*transformation_matrix[6]+ input[4*i+3]*transformation_matrix[7];
      input_transformed[4*i+2] = temp*transformation_matrix[8] + input[4*i+1]*transformation_matrix[9] + input[4*i+2]*transformation_matrix[10]+ input[4*i+3]*transformation_matrix[11];
      input_transformed[4*i+3] = temp*transformation_matrix[12] + input[4*i+1]*transformation_matrix[13] + input[4*i+2]*transformation_matrix[14]+ input[4*i+3]*transformation_matrix[15];
  }
}

void determine_correspondence(float *input, float *output,int *size_source, int *size_target, float *result) {
  float max_distance_sqr = (float) 0.0004f;
  int found = 0;

  //TODO : KDSearch
  for (int i = 0 ; i!= *size_source; i++) {
    for (int k = 0; k!= *size_target; k++  ) {
      float dis;

      //TODO : implement this       tree_->nearestKSearch (input_->points[*idx], 1, index, distance);
      //TODO : Review the 0.02f
      if ((dis == calculate_distance(&input[i*3],&output[k*3]))>0.02f) {
        continue;
      }
      //What if it finds more than 1 ?

      //Save correspondence like this : Index of source point - Index-of found Point - distance
      //Add Correspondence to Result
      result[3*found]= (float)i;
      result[3*found+1] =(float)k;
      result[3*found+2] = dis;
      found = found+1;
      //ADD TO Correspondence cloud.
    }
  }
}
//  shift_and_roll_without_sum
//TODO : cannot pass pointer to auxiliary functions
void computeCorrespondencesCL( float4 guess4f, float *input,float *target,float *correspondence_result,int *size_input,int *size_output,float *input_transformed ) {
  bool ident = true;
  int s_input_local = size_input[0];

  //check for identity
  for (int i = 0 ; i < 4 ; i++) {
    for (int k = 0; i < 4 ; k++) {
      if (i == k ) {
        if (guess4f[i*4+k]!= 1.0f) {
          ident = false;
          break;
        }
      }
      else {
        if(guess4f[i*4+k]!= 0.0f) {
          ident = false;
          break;
        }
      }
    }
  }
  //TODO Affine transformations https://en.wikipedia.org/wiki/Transformation_matrix
  //RIGID transformation Definition at line 190 of file transforms.h.
  //https://libpointmatcher.readthedocs.io/en/latest/Transformations/
  if (ident) {
    rigidTransformationCL(s_input_local,input,guess4f, input_transformed);
  }
  else {
    input_transformed = input;
  }

  //TODO: , called it
  /*
    Methode : determine_correspondence (target, source,)
    Called in : global_classification, line 74
    Source : correspondence_estimation.hpp - 113
    */

  float *res;
  determine_correspondence(input_transformed,target, size_input, size_output, correspondence_result);
  //Correspondence Estimation ?
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



//TODO : 18.03 What is actually be done here ?
//  shift_and_roll_without_sum

// TUAN : Line 374 global_classification

__kernel void shiftAndRollWithoutSumLoop(__global float *floatArgs, __global float *initialTranslation, __global float *direction,__global float *model_voxelized, __global float *point_cloud_ptr, __global float *rotation, __global int *model_voxelized_size, __global int *point_cloud_ptr_size, __global float *correspondence_result, __global float *input_transformed) {

  __private int angle = get_global_id(0);
  __private int shift = get_global_id(1);

  __private float angle_min = floatArgs[0];
  __private float angle_max = floatArgs[1];

  __private float angle_step = floatArgs[2];
  __private float shift_min  = floatArgs[3];

  __private float shift_max = floatArgs[4];
  __private float shift_step = floatArgs[5];


  __private float rot[9];
  __private float trans[3];
  __private float transform[16];

  //CHECKED

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
  __private float shift_temp = shift_min + shift*shift_step;
  trans[0] = initialTranslation[0]*shift_temp/direction[2];
  trans[1] = initialTranslation[1]*shift_temp/direction[2];
  trans[2] = initialTranslation[2]*shift_temp/direction[2];

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

  //computeCorrespondencesCL(transform,model_voxelized,point_cloud_ptr, correspondence_result, model_voxelized_size, point_cloud_ptr_size,input_transformed);

  __private bool ident = true;
  __private int s_input_local = model_voxelized_size[0];
  __private int i = 0;
  __private int k = 0;


  for (i = 0 ; i < 4 ; i++) {
    for (k = 0; i < 4 ; k++) {
      if (i == k ) {
        if (transform[i*4+k]!= 1.0f) {
          ident = false;
        //  break;
        }
      }
      else {
        if(transform[i*4+k]!= 0.0f) {
          ident = false;
        //  break;
        }
      }
    }
  }


  //TODO Affine transformations https://en.wikipedia.org/wiki/Transformation_matrix
  //RIGID transformation Definition at line 190 of file transforms.h.
  //https://libpointmatcher.readthedocs.io/en/latest/Transformations/
  if (ident) {
    //This methode is replaced with following lines
    //rigidTransformationCL(s_input_local,model_voxelized,transform, input_transformed);
    for (int i = 0; i< s_input_local ; i++) {
        float temp = model_voxelized[4*i];
        input_transformed[4*i] = temp*transform[0] + model_voxelized[4*i+1]*transform[1] + model_voxelized[4*i+2]*transform[2]+ model_voxelized[4*i+3]*transform[3];
        input_transformed[4*i+1] = temp*transform[4] + model_voxelized[4*i+1]*transform[5] + model_voxelized[4*i+2]*transform[6]+ model_voxelized[4*i+3]*transform[7];
        input_transformed[4*i+2] = temp*transform[8] + model_voxelized[4*i+1]*transform[9] + model_voxelized[4*i+2]*transform[10]+ model_voxelized[4*i+3]*transform[11];
        input_transformed[4*i+3] = temp*transform[12] + model_voxelized[4*i+1]*transform[13] + model_voxelized[4*i+2]*transform[14]+ model_voxelized[4*i+3]*transform[15];
    }
  }
  else {
    input_transformed = model_voxelized;
  }



  //TODO: , called it
  /*
    Methode : determine_correspondence (target, source,)
    Called in : global_classification, line 74
    Source : correspondence_estimation.hpp - 113

    */

  __private float res[10];
  //This methode is replaced with following lines
  //determine_correspondence(input_transformed,target, size_input, size_output, correspondence_result);
  //TODO : Tuan : 21.03.2018 : This line caused error, reread about calling internal kernel code, or merge these two methods into one

  __private float max_distance_sqr = (float) 0.0004f;
  __private int found = 0;

  //TODO : KDSearch
  for (i = 0 ; i!= *model_voxelized_size; i++) {
    for (k = 0; k!= *point_cloud_ptr_size; k++  ) {
      __private float dis= 0.01f;

      //TODO : implement this       tree_->nearestKSearch (input_->points[*idx], 1, index, distance);
      //TODO : Review the 0.02f
      //if ((dis == calculate_distance(&input_transformed[i*3],&point_cloud_ptr[k*3]))>0.02f) {
      if (dis == 0.02f) {
        continue;
      }
      //What if it finds more than 1 ?

      //Save correspondence like this : Index of source point - Index-of found Point - distance
      //Add Correspondence to Result
      correspondence_result[3*found]= (float)i;
      correspondence_result[3*found+1] =(float)k;
      correspondence_result[3*found+2] = dis;
      found = found+1;
      //ADD TO Correspondence cloud.
    }
  }

}

__kernel void computeDifferencesForCorrespondence(__global float *correspondence_count, __global int *size_correspondence_count,  __global float *angle_count, __global float *shift_count) {
    int i  = get_global_id(0);
    float angle_temp = correspondence_count[i];
    float shift_temp = correspondence_count[i+1];
    float count_temp = correspondence_count[i+2];
    float angleStart;
    float angleStep;
    float angleEnd;
    float shiftStart = 0.0f;
    float shiftEnd = 0.5f;
    float shiftStep = 0.05f;
    float **iterator;
    int iter_help;
    for (int  i = 0; i < size_correspondence_count[0] ; i++) {
      if (angle_count[i]==angle_temp) {
        iter_help = i;
        break;
      }
    }

    if (iter_help != size_correspondence_count[0]) {
      angle_count[iter_help+1] += count_temp;
    } else {
      //Add more into angle_count
      //TODO : Size angle count here is not defined
      //angle_count. //TODO :
    }

    for (int  i = 0; i < size_correspondence_count[0] ; i++) {
      if (shift_count[i]==shift_temp) {
        iter_help = i;
        break;
      }
    }

    if (iter_help != size_correspondence_count[0]) {
      shift_count[iter_help+1] += shift_temp;
    } else {
      //angle_count.push_back //TODO :
    }
    //TODO : Which size here
    int max_index_angles= findMaxIndexOfVectorOfPairsCL(angle_count,size_correspondence_count);
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
}

//https://stackoverflow.com/questions/7627098/what-is-a-lambda-expression-in-c11




//http://docs.pointclouds.org/1.7.0/classpcl_1_1registration_1_1_correspondence_estimation.html

//https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/impl/correspondence_estimation.hpp

//TODO : this one here should be parralized too
