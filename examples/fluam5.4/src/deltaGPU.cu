// Filename: deltaGPU.cu
//
// Copyright (c) 2010-2013, Florencio Balboa Usabiaga
//
// This file is part of Fluam
//
// Fluam is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Fluam is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Fluam. If not, see <http://www.gnu.org/licenses/>.


__device__ void delta4pt2GPU(double x, double &dlx0, double &dlx1, double &dlx2, double &dlx3, double &dlx4){

  if(x<0){
    x = x + dxGPU;
    double s = 0.125 * (3 - 2*x + sqrtf(1 + 4*x*(1-x)) );
    dlx0 = 0.25 * (3-2*x) - s;
    dlx1 = s;
    dlx2 = 0.25 * (2*x-1) + s;
    dlx3 = 0.5 - s;
    dlx4 = 0;
  }
  else{
    double s = 0.125 * (3 - 2*x + sqrtf(1 + 4*x*(1-x)) );
    dlx0 = 0;
    dlx1 = 0.25 * (3-2*x) - s;
    dlx2 = s;
    dlx3 = 0.25 * (2*x-1) + s;
    dlx4 = 0.5 - s;
  }
  
  return;
}

__device__ void delta4ptGPU(const double x, double &dlx0, double &dlx1, double &dlx2, double &dlx3){

  double s = 0.125 * (3 - 2*x + sqrtf(1 + 4*x*(1-x)) );

  dlx0 = 0.25 * (3-2*x) - s;
  dlx1 = s;
  dlx2 = 0.25 * (2*x-1) + s;
  dlx3 = 0.5 - s;
  return;
}



__device__ double delta(double x){
  x = fabs(x);
  if(x > 1.5){
    return 0.;
  }
  else if(x < 0.5){
    return (1. + sqrtf(1. - 3.*x*x))/3.;
  }
  else{
    return (5. - 3. * x - sqrtf(1. - 3. * (1. - x)*(1. - x)))/6.;
  }
}


texture<float, 1, cudaReadModeElementType> texDelta; 
texture<double, 1, cudaReadModeElementType> texDeltaPBC; 

void initDelta(){
  texDelta.normalized = true;
  texDelta.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texDelta.filterMode = cudaFilterModeLinear;//cudaFilterModeLinear and cudaFilterModePoint
  float *h_data;
  int size = 1024;
  h_data = new float[size];
  float x, dx;
  dx = 1.5/float(size);
  x = 0.5 * dx;
  for(int i=0;i<size;i++){
    if(x > 1.5){
      h_data[i] = 0;
    }
    else if(x < 0.5){
      h_data[i] = (1. + sqrtf(1. - 3.*x*x))/3.;
    }
    else{
      h_data[i] = (5. - 3. * x - sqrtf(1. - 3. * (1. - x)*(1. - x)))/6.;
    }
    x += dx;
  }
  h_data[size-1] = 0.;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cutilSafeCall( cudaMallocArray( &cuArrayDelta, &channelDesc, size, 1 )); 
  cutilSafeCall( cudaMemcpyToArray( cuArrayDelta, 0, 0, h_data, size*sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall( cudaBindTextureToArray( texDelta, cuArrayDelta, channelDesc));
  delete[] h_data;
  //cutilSafeCall( cudaUnbindTexture(texDelta) );
  //cutilSafeCall( cudaFreeArray(cuArrayDelta) );
  cout << "INIT DELTA TEXTURE" << endl;
}

bool freeDelta(){
  cutilSafeCall( cudaUnbindTexture(texDelta) );
  cutilSafeCall( cudaFreeArray(cuArrayDelta) );
  cout << "FREE MEMORY DELTA FUNCTION" << endl;
  return 1;
}





