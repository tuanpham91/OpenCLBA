M_PI = 3.141592;
//The idea is to represent row or columns of
kernel void rotateWithCL(float angleInDegrees, float** res) {
  float angle = angleInDegrees * M_PI/180.0f;
  res[0][0] = cos(angle);
  res[0][1] = -sin(angle);
  res[0][2] = 0;
  res[1][0] = sin(angle);
  res[1][1] = cos(angle);
  res[1][2] = 0;
  res[2][0] = 0;
  res[2][1] = 0;
  res[2][2] = 1;
}

//  shift_and_roll_without_sum
__kernel float3 shiftByValueCL(__global float shift, __global float *currentTranslation, __global float* direction ) {
  //TODO : Pass size of currentTranslation


  float v1 = currentTranslation[0]*shift/direction[2];
  float v2 = currentTranslation[1]*shift/direction[2];
  float v3 = currentTranslation[2]*shift/direction[2];
  return (float4)(v1,v2,v3);
}
/*
__kernel void buildTransformationMatrixCL(__global float *rotationDim1, __global float *rotationDim2, __global float *rotationDim3, __global float *translation ) {
  //TODO :

}
*/
//https://stackoverflow.com/questions/36410745/how-to-pass-c-vector-of-vectors-to-opencl-kernel

//  shift_and_roll_without_sum
__kernel void buildTransformationMatrixCL(float **rotation, float3 *translation, float **transformation ) {
//TODO : Is this fast ? Access global memory ?
  transformation[0][0] = rotation[0][0];
  transformation[0][1] = rotation[0][1];
  transformation[0][2] = rotation[0][2];
  transformation[1][0] = rotation[1][0];
  transformation[1][1] = rotation[1][1];
  transformation[1][2] = rotation[1][2];
  transformation[2][0] = rotation[2][0];
  transformation[2][1] = rotation[2][1];
  transformation[2][2] = rotation[2][2];
  transformation[0][3] = translation[0];
  transformation[1][3] = translation[1];
  transformation[2][3] = translation[2];
  transformation[3][0] = 0;
  transformation[3][1] = 0;
  transformation[3][2] = 0;
  transformation[3][3] = 1;
}

//  shift_and_roll_without_sum
__kernel void computeCorrespondencesCL(__global float **guess4f, __global float **input, __global float **target ) {
  //TODO : Best way to send a matrix
  float **input_transformed;
  bool ident = true;
  //check for identity
  for (int i = 0 ; i < 4 ; i++) {
    for (int k = 0; i < 4 ; k++) {
      if (i == k ) {
        if (guess4f[i][k]!= 1) {
          ident = false;
          break;
        }
      }
      else {
        if(guess4f[i][k]!= 0) {
          ident = false;
          break
        }
      }
    }
  }
  //TODO Affine transformations https://en.wikipedia.org/wiki/Transformation_matrix
  //RIGID transformation Definition at line 190 of file transforms.h.
  //https://libpointmatcher.readthedocs.io/en/latest/Transformations/
  if (ident) {
    rigidTransformation(guess4f,input,input_transformed);
  }
  else {
    input_transformed = input;
  }
  //Correspondence Estimation ?
}


__kernel void rigidTransformation (__global int size, __global float **input, __global float **transformation_matrix) {
  for (int i = 0; i< size ; i++) {
      float temp = input[i][0];
      input[i][0] = temp*transformation_matrix[0][0] + input[i][1]*transformation_matrix[0][1] + input[i][2]*transformation_matrix[0][2]+ + input[i][3]*transformation_matrix[0][3];
      input[i][1] = temp*transformation_matrix[1][0] + input[i][1]*transformation_matrix[1][1] + input[i][2]*transformation_matrix[1][2]+ + input[i][3]*transformation_matrix[1][3];
      input[i][2] = temp*transformation_matrix[2][0] + input[i][1]*transformation_matrix[2][1] + input[i][2]*transformation_matrix[2][2]+ + input[i][3]*transformation_matrix[2][3];
      input[i][3] = temp*transformation_matrix[3][0] + input[i][1]*transformation_matrix[3][1] + input[i][2]*transformation_matrix[3][2]+ + input[i][3]*transformation_matrix[3][3];
  }
}
//  shift_and_roll_without_sum
// NOTE : This might not be useful :
__kernel void compute


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
__kernel void findMaxIndexOfVectorOfPairsCL(__global float **angle_count, __global int *size, global int *res) {
  int max_index =0;
  float max= 0.0f;
  for (int i = 0 ; i < size ; i++) {
    if (angle_count[i][1]>max) {
      max = angle_count[i][1];
      max_index = i;
    }
  }
  res = max_index;
}

__kernel void findMaxIndexOfVectorOfTuplesCL(__global float **tuples, __global int size. global int *res) {
  int max_index =0;
  float max= 0.0f;
  for (int i = 0 ; i < size ; i++) {
    if (angle_count[i][2]>max) {
      max = angle_count[i][2];
      max_index = i;
    }
  }
  res = max_index;
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

//http://docs.pointclouds.org/1.7.0/classpcl_1_1registration_1_1_correspondence_estimation.html

//https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/impl/correspondence_estimation.hpp

//TODO : this one here should be parralized too
void estimate_correspondence( float **input, float **output, float max_distance, float size_source, float size_target, float **result) {
  double max_distance_sqr = max_distance*max_distance;
  found = 0;

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


double calculate_distance(float *pointA, float *pointB)  {
  double a = (pointA[0] - pointB[0])*(pointA[0] - pointB[0]);
  double b = (pointA[1] - pointB[1])*(pointA[1] - pointB[1]);
  double c = (pointA[2] - pointB[2])*(pointA[2] - pointB[2]);
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


__kernel void shift_and_roll_without_sum_loop(__global float* floatArgs, __global float* count, __global float* initialTranslation, __global float* direction,
                                              __global float** model_voxelized, __global float** point_cloud_ptr, __global float **modelTransformed) {}
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

  rotateWithCL(angle_min+ angle*angle_step, rot, rotated);
  float3 trans = shiftByValueCL(shift_min+ shift*shift_step, initialTranslation, direction );
  buildTransformationMatrixCL(rotated,&trans,transform);

  //TODO : Assert this
  count = computeCorrespondencesCL(transform,model_voxelized,point_cloud_ptr );

}
