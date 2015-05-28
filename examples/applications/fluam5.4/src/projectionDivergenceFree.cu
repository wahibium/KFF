// Filename: projectionDivergenceFree.cu
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


__global__ void projectionDivergenceFree(cufftDoubleComplex *vxZ,
					 cufftDoubleComplex *vyZ,
					 cufftDoubleComplex *vzZ,
					 prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;

  //Construct GG
  double GG;
  GG = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));
  

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = pF->gradKx[kx].y * vxZ[i].x + pF->gradKy[ky].y * vyZ[i].x + pF->gradKz[kz].y * vzZ[i].x;
  GW.y = pF->gradKx[kx].y * vxZ[i].y + pF->gradKy[ky].y * vyZ[i].y + pF->gradKz[kz].y * vzZ[i].y;



  if(i==0){
    vxZ[i].x = vxZ[i].x;
    vxZ[i].y = vxZ[i].y;
    vyZ[i].x = vyZ[i].x;
    vyZ[i].y = vyZ[i].y;
    vzZ[i].x = vzZ[i].x;
    vzZ[i].y = vzZ[i].y;
  }
  else{
    vxZ[i].x = (vxZ[i].x + pF->gradKx[kx].y * GW.x / GG) ;
    vxZ[i].y = (vxZ[i].y + pF->gradKx[kx].y * GW.y / GG) ;
    vyZ[i].x = (vyZ[i].x + pF->gradKy[ky].y * GW.x / GG) ;
    vyZ[i].y = (vyZ[i].y + pF->gradKy[ky].y * GW.y / GG) ;
    vzZ[i].x = (vzZ[i].x + pF->gradKz[kz].y * GW.x / GG) ;
    vzZ[i].y = (vzZ[i].y + pF->gradKz[kz].y * GW.y / GG) ;
  }


  




}



















__global__ void projectionDivergenceFree2D(cufftDoubleComplex *vxZ,
					   cufftDoubleComplex *vyZ,
					   cufftDoubleComplex *vzZ,
					   prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int kx, ky;
  //kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;

  //Construct GG
  double GG;
  GG = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) ;
  

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = pF->gradKx[kx].y * vxZ[i].x + pF->gradKy[ky].y * vyZ[i].x ;
  GW.y = pF->gradKx[kx].y * vxZ[i].y + pF->gradKy[ky].y * vyZ[i].y ;



  if(i==0){
    vxZ[i].x = vxZ[i].x;
    vxZ[i].y = vxZ[i].y;
    vyZ[i].x = vyZ[i].x;
    vyZ[i].y = vyZ[i].y;
  }
  else{
    vxZ[i].x = (vxZ[i].x + pF->gradKx[kx].y * GW.x / GG) ;
    vxZ[i].y = (vxZ[i].y + pF->gradKx[kx].y * GW.y / GG) ;
    vyZ[i].x = (vyZ[i].x + pF->gradKy[ky].y * GW.x / GG) ;
    vyZ[i].y = (vyZ[i].y + pF->gradKy[ky].y * GW.y / GG) ;
  }


  




}
