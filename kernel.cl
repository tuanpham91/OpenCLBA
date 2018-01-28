M_PI = 3.141592;
//The idea is to represent row or columns of
__kernel void * rotateWithCL(__global float angleInDegrees, __global float *matrix1, __global float *matrix2, __global float *matrix3, __global float *res) {
  res[0] = cos(angle);
  float angle = angleInDegrees * M_PI/180.0f;
  res[1] = -sin(angle);
  res[2] = 0;
  res[3] = sin(angle);
  res[4] = cos(angle);
  res[5] = 0;
  res[6] = 0;
  res[7] = 0;
  res[8] = 1;
}

//  shift_and_roll_without_sum
__kernel void shiftByValueCL(__global float shift, __global float *currentTranslation, __global float direction ) {
  //TODO : Pass size of currentTranslation
  currentTranslation = currentTranslation*shift/direction;
}
/*
__kernel void buildTransformationMatrixCL(__global float *rotationDim1, __global float *rotationDim2, __global float *rotationDim3, __global float *translation ) {
  //TODO :

}
*/
//https://stackoverflow.com/questions/36410745/how-to-pass-c-vector-of-vectors-to-opencl-kernel

//  shift_and_roll_without_sum
__kernel void buildTransformationMatrixCL(__global float **rotation, __global float *translation, __global float **transformation ) {
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
__kernel void computeCorrespondencesCL(__global std::vector<std::vector<float>> guess, __global float ****input, __global ****target, ****target  ) {
  //TODO : Best way to send a matrix
}

//  shift_and_roll_without_sum
// NOTE : This might not be useful :
__kernel void compute
