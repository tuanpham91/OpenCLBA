void rotateWithCL(float angleInDegrees, float* res) {
  float angle = angleInDegrees*(float)(0.01745328888);
  res[0] = cos(angle);
  res[1] = -sin(angle);
  res[2] = 0;
  res[3] = sin(angle);
  res[4] = cos(angle);
  res[5] = 0;
  res[6] = 0;
  res[7] = 0;
  res[6] = 1;
}

//  shift_and_roll_without_sum
float3 shiftByValueCL(float shift,float *currentTranslation,float *direction ) {
  float v1 = currentTranslation[0]*shift/direction[2];
  float v2 = currentTranslation[1]*shift/direction[2];
  float v3 = currentTranslation[2]*shift/direction[2];
  return (float3)(v1,v2,v3);
}

//https://stackoverflow.com/questions/36410745/how-to-pass-c-vector-of-vectors-to-opencl-kernel

//  shift_and_roll_without_sum
void buildTransformationMatrixCL(float *rotation, float3 translation, float *transformation ) {
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

void rigidTransformationCL (int size,float *input,float *transformation_matrix) {
  for (int i = 0; i< size ; i++) {
      float temp = input[4*i];
      input[4*i] = temp*transformation_matrix[0] + input[4*i+1]*transformation_matrix[1] + input[4*i+2]*transformation_matrix[2]+ input[4*i+3]*transformation_matrix[3];
      input[4*i+1] = temp*transformation_matrix[4] + input[4*i+1]*transformation_matrix[5] + input[4*i+2]*transformation_matrix[6]+ input[4*i+3]*transformation_matrix[7];
      input[4*i+2] = temp*transformation_matrix[8] + input[4*i+1]*transformation_matrix[9] + input[4*i+2]*transformation_matrix[10]+ input[4*i+3]*transformation_matrix[11];
      input[4*i+3] = temp*transformation_matrix[12] + input[4*i+1]*transformation_matrix[13] + input[4*i+2]*transformation_matrix[14]+ input[4*i+3]*transformation_matrix[15];
  }
}
//  shift_and_roll_without_sum
__kernel void computeCorrespondencesCL(__global float *guess4f, __global float *input, __global float *target ) {
  //TODO : Best way to send a matrix
  float *input_transformed;
  bool ident = true;
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
    rigidTransformationCL(guess4f,input,input_transformed);
  }
  else {
    input_transformed = input;
  }
  //TODO
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

//  shift_and_roll_without_sum
__kernel void computeDifferencesForCorrespondence(__global float **correspondence_count, __global int size, __global int **angle_count,) {
    int i  = get_global_id(0);
    float angle_temp = correspondence_count[i][0];
    float shift_temp = correspondence_count[i][1];
    float count_temp = correspondence_count[i][2];
    float angleStart;
    float angleStep;
    float angleEnd;
    float shiftStart = 0.0f;
    float shiftEnd = 0.5;
    float shiftStep = 0.05f;
    float **iterator;
    int iter_help;
    for (int  i = 0; i < size ; i++) {
      if (angle_count[i][0]==angle_temp) {
        iter_help = i;
        break;
      }
    }

    if (iter_help != size) {
      angle_count[iter_help][1] += count_temp;
    } else {
      angle_count.push_back //TODO :
    }

    for (int  i = 0; i < size ; i++) {
      if (shift_count[i][0]==shift_temp) {
        iter_help = i;
        break;
      }
    }

    if (iter_help != size) {
      shift_count[iter_help][1] += shift_temp;
    } else {
      angle_count.push_back //TODO :
    }

    int max_index_angles = findMaxIndexOfVectorOfPairsCL(angle_count);
    int max_index_shift = findMaxIndexOfVectorOfPairsCL(shift_count);
    int correspondence_index == findMaxIndexOfVectorOfTuplesCL(correspondence_count);

    ;
    float max_angle = angle_count[max_index_angles][0];
    float max_shift = angle_count[max_index_angles][0];

    angleStart = checkMinBoundsForValueCL(max_angle,angleStart,angleStep);
    angleEnd = checkMaxBoundsForValueCL(max_angle,angleEnd, angleStep);
    shiftStart = checkMinBoundsForValueCL(max_shift,shiftStart,shiftStep);
    shiftEnd = checkMaxBoundsForValueCL(max_shift,shiftEnd,shiftStep);
    angleStep /= 5.0f;
    shiftStep /= 5.0f
    //TODO :
}

//https://stackoverflow.com/questions/7627098/what-is-a-lambda-expression-in-c11
__kernel void findMaxIndexOfVectorOfPairsCL(__global float *angle_count, __global int *size,__global int *res) {
  int max_index =0;
  float max= 0.0f;
  for (int i = 0 ; i < *size ; i++) {
    if (angle_count[3*i+1]>max) {
      max = angle_count[i*3+1];
      max_index = i;
    }
  }
  *res = max_index;
}

__kernel void findMaxIndexOfVectorOfTuplesCL(__global float *tuples, __global int size,__global int *res) {
  int max_index =0;
  float max= 0.0f;
  //TODO : Review this
  for (int i = 0 ; i < size ; i++) {
    if (tuples[i*3+2]>max) {
      max = tuples[i*3+2];
      max_index = i;
    }
  }
  *res = max_index;
}


//http://docs.pointclouds.org/1.7.0/classpcl_1_1registration_1_1_correspondence_estimation.html

//https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/impl/correspondence_estimation.hpp

//TODO : this one here should be parralized too
void estimate_correspondence(float *input, float *output, float max_distance, float size_source, float size_target, float *result) {
  float max_distance_sqr = (float)max_distance*max_distance;
  int found = 0;
  for (int i = 0 ; i!= size_source; i++) {
    for (int k = 0; k!= size_target; k++  ) {
      if (calculate_distance(input[i],output[k])>max_distance) {
        continue;
      }
      //What if it finds more than 1 ?
      result[found]= output[k];
      found = found+1;
      break;
      //ADD TO Correspondence cloud.
    }
  }
}


float calculate_distance(float *pointA, float *pointB)  {
  float a = (pointA[0] - pointB[0])*(pointA[0] - pointB[0]);
  float b = (pointA[1] - pointB[1])*(pointA[1] - pointB[1]);
  float c = (pointA[2] - pointB[2])*(pointA[2] - pointB[2]);
  return sqrt(a+b+c);
}

/*

__kernel void shift_and_roll_without_sum_loop(__global float* initialTranslation, __global float* shift, __global float* angle_min,
                                              __global float* angle, __global float** rotation, __global float* trans,
                                              __global float** transform, __global float* correspondence_count,
                                              __global float* direction, __global float* angle_max, __global float* shift_min,
                                              __global float* shift_max, __global float** model_voxelized, __global float** point_cloud_ptr
                                              __global float** model_transformed) {
*/



__kernel void shift_and_roll_without_sum_loop(__global float* floatArgs, __global float* count, __global float* initialTranslation, __global float* direction,__global float* model_voxelized, __global float* point_cloud_ptr, __global float *rotation) {
  int angle = get_global_id(0);
  int shift = get_global_id(1);

  float angle_min = floatArgs[0];
  float angle_max = floatArgs[1];

  float angle_step = floatArgs[2];
  float shift_min  = floatArgs[3];

  float shift_max = floatArgs[4];
  float shift_step = floatArgs[5];


  float **rotated;
  float **transform;
  //TODO
  rotateWithCL(angle_min+ angle*angle_step, rotation, rotated);
//TODO
  float3 trans = shiftByValueCL(shift_min+ shift*shift_step, initialTranslation, direction );
  buildTransformationMatrixCL(rotated,&trans,transform);

  //TODO : Assert this
  count = computeCorrespondencesCL(transform,model_voxelized,point_cloud_ptr );
}
