// Filename: kernelUpdateVIncompressible.cu
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


__global__ void kernelUpdateVIncompressible(cufftDoubleComplex *vxZ, 
					    cufftDoubleComplex *vyZ,
					    cufftDoubleComplex *vzZ, 
					    cufftDoubleComplex *WxZ, 
					    cufftDoubleComplex *WyZ, 
					    cufftDoubleComplex *WzZ, 
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

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = pF->gradKx[kx].y * WxZ[i].x + pF->gradKy[ky].y * WyZ[i].x + pF->gradKz[kz].y * WzZ[i].x;
  GW.y = pF->gradKx[kx].y * WxZ[i].y + pF->gradKy[ky].y * WyZ[i].y + pF->gradKz[kz].y * WzZ[i].y;
  
  if(i==0){
    vxZ[i].x = WxZ[i].x;
    vxZ[i].y = WxZ[i].y;
    vyZ[i].x = WyZ[i].x;
    vyZ[i].y = WyZ[i].y;
    vzZ[i].x = WzZ[i].x;
    vzZ[i].y = WzZ[i].y;
  }
  else{
    vxZ[i].x = (WxZ[i].x + pF->gradKx[kx].y * GW.x / GG) / denominator;
    vxZ[i].y = (WxZ[i].y + pF->gradKx[kx].y * GW.y / GG) / denominator;
    vyZ[i].x = (WyZ[i].x + pF->gradKy[ky].y * GW.x / GG) / denominator;
    vyZ[i].y = (WyZ[i].y + pF->gradKy[ky].y * GW.y / GG) / denominator;
    vzZ[i].x = (WzZ[i].x + pF->gradKz[kz].y * GW.x / GG) / denominator;
    vzZ[i].y = (WzZ[i].y + pF->gradKz[kz].y * GW.y / GG) / denominator;
  }
  
}

// ==================================
// A. Donev
// This uses Backward Euler instead of Crank-Nicolson method
// ==================================

__global__ void kernelUpdateVIncompressibleBE(cufftDoubleComplex *vxZ, cufftDoubleComplex *vyZ,
					      cufftDoubleComplex *vzZ, cufftDoubleComplex *WxZ, 
					      cufftDoubleComplex *WyZ, cufftDoubleComplex *WzZ, 
					      prefactorsFourier *pF, double identity_prefactor){
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
  double denominator = identity_prefactor - dtGPU * shearviscosityGPU * L / densfluidGPU;

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = pF->gradKx[kx].y * WxZ[i].x + pF->gradKy[ky].y * WyZ[i].x + pF->gradKz[kz].y * WzZ[i].x;
  GW.y = pF->gradKx[kx].y * WxZ[i].y + pF->gradKy[ky].y * WyZ[i].y + pF->gradKz[kz].y * WzZ[i].y;
  
  if(i==0){
    vxZ[i].x = WxZ[i].x;
    vxZ[i].y = WxZ[i].y;
    vyZ[i].x = WyZ[i].x;
    vyZ[i].y = WyZ[i].y;
    vzZ[i].x = WzZ[i].x;
    vzZ[i].y = WzZ[i].y;
  }
  else{
    vxZ[i].x = (WxZ[i].x + pF->gradKx[kx].y * GW.x / GG) / denominator;
    vxZ[i].y = (WxZ[i].y + pF->gradKx[kx].y * GW.y / GG) / denominator;
    vyZ[i].x = (WyZ[i].x + pF->gradKy[ky].y * GW.x / GG) / denominator;
    vyZ[i].y = (WyZ[i].y + pF->gradKy[ky].y * GW.y / GG) / denominator;
    vzZ[i].x = (WzZ[i].x + pF->gradKz[kz].y * GW.x / GG) / denominator;
    vzZ[i].y = (WzZ[i].y + pF->gradKz[kz].y * GW.y / GG) / denominator;
  }
  
}











__global__ void kernelUpdateVIncompressible2D(cufftDoubleComplex *vxZ, 
					      cufftDoubleComplex *vyZ,
					      cufftDoubleComplex *vzZ, 
					      cufftDoubleComplex *WxZ, 
					      cufftDoubleComplex *WyZ, 
					      cufftDoubleComplex *WzZ, 
					      prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int kx, ky;
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y))  ;

  //Construct GG
  double GG;
  GG = L;

  //Construct denominator
  double denominator = 1 - 0.5 * dtGPU * shearviscosityGPU * L / densfluidGPU;

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = pF->gradKx[kx].y * WxZ[i].x + pF->gradKy[ky].y * WyZ[i].x ;
  GW.y = pF->gradKx[kx].y * WxZ[i].y + pF->gradKy[ky].y * WyZ[i].y ;
  
  if(i==0){
    vxZ[i].x = WxZ[i].x;
    vxZ[i].y = WxZ[i].y;
    vyZ[i].x = WyZ[i].x;
    vyZ[i].y = WyZ[i].y;
  }
  else{
    vxZ[i].x = (WxZ[i].x + pF->gradKx[kx].y * GW.x / GG) / denominator;
    vxZ[i].y = (WxZ[i].y + pF->gradKx[kx].y * GW.y / GG) / denominator;
    vyZ[i].x = (WyZ[i].x + pF->gradKy[ky].y * GW.x / GG) / denominator;
    vyZ[i].y = (WyZ[i].y + pF->gradKy[ky].y * GW.y / GG) / denominator;
  }
  
}
