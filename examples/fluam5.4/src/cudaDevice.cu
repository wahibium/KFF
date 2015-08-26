// Filename: cudaDevice.cu
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


#include "header.h"

bool cudaDevice(){
  int deviceCount = 0;
  int driverVersion = 0;
  int runtimeVersion = 0;
  cudaDeviceProp deviceProp;
  
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess){
    cout << "cudaGetDeviceCount ERROR" << endl;
    return 0;
  }


  string NombreCuda;
  NombreCuda =  outputname + ".cuda";
  ofstream fileSaveCuda(NombreCuda.c_str());
  //fileSaveCuda.width(50);
  fileSaveCuda << endl << "cudaGetDeviceCount     " << deviceCount << endl;

  for(int i=0;i<deviceCount;i++){
    cudaGetDeviceProperties(&deviceProp, i);
    fileSaveCuda << endl << "Device " << i << ": version " << deviceProp.major << " " << deviceProp.minor << endl;
    fileSaveCuda << deviceProp.name << endl;

    cudaDriverGetVersion(&driverVersion);
    fileSaveCuda << "Driver version                  " << driverVersion << endl;

    cudaRuntimeGetVersion(&runtimeVersion);
    fileSaveCuda << "Runtime version                 " << runtimeVersion << endl;

    fileSaveCuda << "Multiprocessors                 " << deviceProp.multiProcessorCount << endl;

    fileSaveCuda << "Global memory (bytes)           " << deviceProp.totalGlobalMem << endl;
    fileSaveCuda << "Constant memory (bytes)         " << deviceProp.totalConstMem << endl;
    fileSaveCuda << "Shared memory per block (bytes) " << deviceProp.sharedMemPerBlock << endl;
    fileSaveCuda << "Registers per block             " << deviceProp.regsPerBlock << endl;
    fileSaveCuda << "Warp size                       " << deviceProp.warpSize << endl;
    fileSaveCuda << "Maximum threads per block       " << deviceProp.maxThreadsPerBlock << endl;
    fileSaveCuda << "Clock rate (GHz)                " << deviceProp.clockRate*1e-6 << endl;
    fileSaveCuda << "Compute Mode                    " << deviceProp.computeMode << endl;


  }

  if(setDevice>=0){
    int *device;
    device = new int [1];
    cudaSetDevice(setDevice);
    cudaGetDevice(device);
    fileSaveCuda << endl << endl << endl;
    fileSaveCuda << "Device selected " << device[0] << endl;
    delete[] device;
  }


  fileSaveCuda << endl;
  fileSaveCuda.close();



  return 1;
}
