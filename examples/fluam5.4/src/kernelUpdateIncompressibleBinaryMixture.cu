// Filename: kernelUpdateIncompressibleBinaryMixture.cu
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


__global__ void kernelUpdateIncompressibleBinaryMixture(cufftDoubleComplex *vxZ, 
							cufftDoubleComplex *vyZ,
							cufftDoubleComplex *vzZ, 
							cufftDoubleComplex *cZ,
							prefactorsFourier *pF){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));

  //Construct GG
  double GG;
  GG = L;

  //Construct denominator
  double denominator = 1 - 0.5 * dtGPU * shearviscosityGPU * L / densfluidGPU;
  double denominatorConcentration = 1 - 0.5 * dtGPU * diffusionGPU * L ;

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = pF->gradKx[kx].y * vxZ[i].x + pF->gradKy[ky].y * vyZ[i].x + pF->gradKz[kz].y * vzZ[i].x;
  GW.y = pF->gradKx[kx].y * vxZ[i].y + pF->gradKy[ky].y * vyZ[i].y + pF->gradKz[kz].y * vzZ[i].y;
  
  if(i==0){
    //vxZ[i].x = WxZ[i].x;
    //vxZ[i].y = WxZ[i].y;
    //vyZ[i].x = WyZ[i].x;
    //vyZ[i].y = WyZ[i].y;
    //vzZ[i].x = WzZ[i].x;
    //vzZ[i].y = WzZ[i].y;
  }
  else{
    vxZ[i].x = (vxZ[i].x + pF->gradKx[kx].y * GW.x / GG) / denominator;
    vxZ[i].y = (vxZ[i].y + pF->gradKx[kx].y * GW.y / GG) / denominator;
    vyZ[i].x = (vyZ[i].x + pF->gradKy[ky].y * GW.x / GG) / denominator;
    vyZ[i].y = (vyZ[i].y + pF->gradKy[ky].y * GW.y / GG) / denominator;
    vzZ[i].x = (vzZ[i].x + pF->gradKz[kz].y * GW.x / GG) / denominator;
    vzZ[i].y = (vzZ[i].y + pF->gradKz[kz].y * GW.y / GG) / denominator;
    cZ[i].x = cZ[i].x / denominatorConcentration ;
    cZ[i].y = cZ[i].y / denominatorConcentration ;   
  }
  
}


// ==================================
// A. Donev
// Crank-Nicolson for concentration only
// ==================================

__global__ void kernelUpdateIncompressibleBinaryConcentration(cufftDoubleComplex *cZ,
							      prefactorsFourier *pF){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));

  //Construct denominator
  double denominatorConcentration = 1 - 0.5 * dtGPU * diffusionGPU * L ;
  
  if(i==0){
    // Zero mode
  }
  else{
    cZ[i].x = cZ[i].x / denominatorConcentration ;
    cZ[i].y = cZ[i].y / denominatorConcentration ;   
  }
  
}
