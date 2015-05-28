// Filename: initializePrefactorFourierSpace.cu
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


__global__ void initializePrefactorFourierSpace_1(cufftDoubleComplex *gradKx, 
						  cufftDoubleComplex *gradKy,
						  cufftDoubleComplex *gradKz,
						  cufftDoubleComplex *expKx,
						  cufftDoubleComplex *expKy,
						  cufftDoubleComplex *expKz,
						  prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>0) return;   

  pF->gradKx = gradKx;
  pF->gradKy = gradKy;
  pF->gradKz = gradKz;
  pF->expKx = expKx;
  pF->expKy = expKy;
  pF->expKz = expKz;

}



__global__ void initializePrefactorFourierSpace_2(prefactorsFourier *pF){
  
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  double pi = 4. * atan(1.);
  
  if(j<mxGPU){
    pF->gradKx[j].x = 0;
    pF->gradKx[j].y = 2 * invdxGPU * sin(pi*j/double(mxGPU));
    pF->expKx[j].x = cos(pi*j/double(mxGPU));
    pF->expKx[j].y = sin(pi*j/double(mxGPU));
  }

  if(j<myGPU){
    pF->gradKy[j].x = 0;
    pF->gradKy[j].y = 2 * invdyGPU * sin(pi*j/double(myGPU));
    pF->expKy[j].x = cos(pi*j/double(myGPU));
    pF->expKy[j].y = sin(pi*j/double(myGPU));
  }

  if(j<mzGPU){
    pF->gradKz[j].x = 0;
    pF->gradKz[j].y = 2 * invdzGPU * sin(pi*j/double(mzGPU));
    pF->expKz[j].x = cos(pi*j/double(mzGPU));
    pF->expKz[j].y = sin(pi*j/double(mzGPU));
  }

}




