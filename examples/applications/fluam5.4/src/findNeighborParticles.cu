// Filename: findNeighborParticles.cu
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



__global__ void countPartInCellNonBonded_to_zero(particlesincell* pc){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=mNeighborsGPU) return; 
  
  pc->countPartInCellNonBonded[i] = 0;
}



__global__ void findNeighborParticles(particlesincell* pc, int* errorKernel,
				      double* rxboundaryGPU, double* ryboundaryGPU, double* rzboundaryGPU,
				      double* vxboundaryGPU, double* vyboundaryGPU, double* vzboundaryGPU){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU+nboundaryGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,i);
  double ry = fetch_double(texryboundaryGPU,i);
  double rz = fetch_double(texrzboundaryGPU,i);

  if(i>=nboundaryGPU){
    rx += vxboundaryGPU[i] * dtGPU;
    ry += vyboundaryGPU[i] * dtGPU;
    rz += vzboundaryGPU[i] * dtGPU;
    rxboundaryGPU[i] = rx;
    ryboundaryGPU[i] = ry;
    rzboundaryGPU[i] = rz;
    //printf("%i %f %f %f \n",i,rx,ry,rz);
  }
  //double r;
  int icel;
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    //r = rx;
    rx = rx - (int(rx*invlxGPU + 0.5*((rx>0)-(rx<0)))) * lxGPU;
    int jx   = int(rx * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    //r = ry;
    ry = ry - (int(ry*invlyGPU + 0.5*((ry>0)-(ry<0)))) * lyGPU;
    int jy   = int(ry * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    //r = rz;
    rz = rz - (int(rz*invlzGPU + 0.5*((rz>0)-(rz<0)))) * lzGPU;
    int jz   = int(rz * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
    //printf("UUUUUU %i %f\n",jz,rz);
    //printf("PART %i CELL %i X %f %i Y %f %i Z %f %i\n",i,icel,rx,jx,ry,jy,rz,jz);
  }
  int np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  //printf("KKKKKKK %i %i %i\n",i,icel,np);
  //printf("XXXX %i %i %i %i \n",mNeighborsGPU,np,icel,i);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  //printf("gGGGGGGG  %i %i %i %i\n",np,i,icel,mNeighborsGPU*np+icel);
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;
  //printf("KKKKKKKK  %i %i %i %i\n",np,i,icel,mNeighborsGPU*np+icel);

}
