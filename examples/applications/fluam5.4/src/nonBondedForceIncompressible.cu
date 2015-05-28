// Filename: nonBondedForceIncompressible.cu
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


__global__ void nonBondedForceIncompressible(double* rxcellGPU, 
					     double* rycellGPU, 
					     double* rzcellGPU,
					     double* vxboundaryGPU, 
					     double* vyboundaryGPU, 
					     double* vzboundaryGPU,
					     double* densityGPU,
					     double* fxboundaryGPU, 
					     double* fyboundaryGPU, 
					     double* fzboundaryGPU,
					     particlesincell* pc, 
					     int* errorKernel,
					     double* saveForceX, 
					     double* saveForceY, 
					     double* saveForceZ){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double fx = 0.;
  double fy = 0.;
  double fz = 0.;
  double f;

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);


  double rxij, ryij, rzij, r2;

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;
  
  int icel;
  double r, rp, rm;
  
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }
  
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellNonBonded,icel);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+icel);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }  
  //Particles in Cell vecino0
  vecino0 = tex1Dfetch(texneighbor0GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino0);
  //printf("RRRRRRRRR %i %i %i\n",i,icel,vecino0);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino0);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino1
  vecino1 = tex1Dfetch(texneighbor1GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino1);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino1);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
   }
  //Particles in Cell vecino2
  vecino2 = tex1Dfetch(texneighbor2GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino2);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino2);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino3
  vecino3 = tex1Dfetch(texneighbor3GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino3);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino3);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino4
  vecino4 = tex1Dfetch(texneighbor4GPU, icel);
  //printf("VECINO %i %i \n",icel,vecino4);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino4);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino4);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino5
  vecino5 = tex1Dfetch(texneighbor5GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino5);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino5);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpy
  vecinopxpy = tex1Dfetch(texneighborpxpyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpy);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmy
  vecinopxmy = tex1Dfetch(texneighborpxmyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmy);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpz
  vecinopxpz = tex1Dfetch(texneighborpxpzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmz
  vecinopxmz = tex1Dfetch(texneighborpxmzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpy
  vecinomxpy = tex1Dfetch(texneighbormxpyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpy);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij; 
  }
  //Particles in Cell vecinomxmy
  vecinomxmy = tex1Dfetch(texneighbormxmyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmy);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpz
  vecinomxpz = tex1Dfetch(texneighbormxpzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxmz
  vecinomxmz = tex1Dfetch(texneighbormxmzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopypz
  vecinopypz = tex1Dfetch(texneighborpypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopymz
  vecinopymz = tex1Dfetch(texneighborpymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
   //Particles in Cell vecinomypz
  vecinomypz = tex1Dfetch(texneighbormypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomymz
  vecinomymz = tex1Dfetch(texneighbormymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpypz
  vecinopxpypz = tex1Dfetch(texneighborpxpypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpymz
  vecinopxpymz = tex1Dfetch(texneighborpxpymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmypz
  vecinopxmypz = tex1Dfetch(texneighborpxmypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmymz
  vecinopxmymz = tex1Dfetch(texneighborpxmymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpypz
  vecinomxpypz = tex1Dfetch(texneighbormxpypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpymz
  vecinomxpymz = tex1Dfetch(texneighbormxpymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxmypz
  vecinomxmypz = tex1Dfetch(texneighbormxmypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxmymz
  vecinomxmymz = tex1Dfetch(texneighbormxmymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  
    
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * myGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }

  np = atomicAdd(&pc->countparticlesincellX[icelx],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellsGPU*np+icelx] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellY[icely],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[2]=np;
    return;
  }
  pc->partincellY[ncellsGPU*np+icely] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellZ[icelz],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[3]=np;
    return;
  }
  pc->partincellZ[ncellsGPU*np+icelz] = nboundaryGPU+i;

  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;


  //FORCE IN THE X DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icelx);
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecino5 = tex1Dfetch(texvecino5GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelx);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelx);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelx);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelx);
  int vecinopxpxpypz = tex1Dfetch(texvecino3GPU, vecinopxpypz);
  int vecinopxpxpymz = tex1Dfetch(texvecino3GPU, vecinopxpymz);
  int vecinopxpxmypz = tex1Dfetch(texvecino3GPU, vecinopxmypz);
  int vecinopxpxmymz = tex1Dfetch(texvecino3GPU, vecinopxmymz);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  int vecinopxpxpz   = tex1Dfetch(texvecino3GPU, vecinopxpz);
  int vecinopxpxmz   = tex1Dfetch(texvecino3GPU, vecinopxmz);


  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = tex1D(texDelta, fabs(r));
  dlxp = tex1D(texDelta, fabs(rp));
  dlxm = tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = tex1D(texDelta, fabs(r));
  dlyp = tex1D(texDelta, fabs(rp));
  dlym = tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = tex1D(texDelta, fabs(r));
  dlzp = tex1D(texDelta, fabs(rp));
  dlzm = tex1D(texDelta, fabs(rm));

  double v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);

  double rho = dlxm * dlym * dlzm +
    dlxm * dlym * dlz  +
    dlxm * dlym * dlzp +
    dlxm * dly  * dlzm + 
    dlxm * dly  * dlz  +
    dlxm * dly  * dlzp +
    dlxm * dlyp * dlzm +
    dlxm * dlyp * dlz  +
    dlxm * dlyp * dlzp +
    dlx  * dlym * dlzm +
    dlx  * dlym * dlz  +
    dlx  * dlym * dlzp + 
    dlx  * dly  * dlzm + 
    dlx  * dly  * dlz  + 
    dlx  * dly  * dlzp + 
    dlx  * dlyp * dlzm + 
    dlx  * dlyp * dlz  + 
    dlx  * dlyp * dlzp + 
    dlxp * dlym * dlzm +
    dlxp * dlym * dlz  + 
    dlxp * dlym * dlzp +
    dlxp * dly  * dlzm + 
    dlxp * dly  * dlz  + 
    dlxp * dly  * dlzp +
    dlxp * dlyp * dlzm +
    dlxp * dlyp * dlz  +
    dlxp * dlyp * dlzp ;


  double u, uold, fact;

  uold = vxboundaryGPU[nboundaryGPU+i];
  fact = 1.5 * volumeParticleGPU * volumeGPU * densfluidGPU * rho;
  u = uold + (fact * (v-uold) + fx*dtGPU) / (massParticleGPU + fact);
  vxboundaryGPU[nboundaryGPU+i] = u;

  f = (v - u) * 1.5 * volumeParticleGPU;

  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;


  //FORCE IN THE Y DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icely);
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecino5 = tex1Dfetch(texvecino5GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icely);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icely);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icely);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icely);  
  //DEFINE MORE NEIGHBORS
  int vecinopymxpymz = tex1Dfetch(texvecino4GPU, vecinomxpymz);
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopymxpypz = tex1Dfetch(texvecino4GPU, vecinomxpypz);
  int vecinopypymz   = tex1Dfetch(texvecino4GPU, vecinopymz);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypypz   = tex1Dfetch(texvecino4GPU, vecinopypz);
  int vecinopypxpymz = tex1Dfetch(texvecino4GPU, vecinopxpymz);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);
  int vecinopypxpypz = tex1Dfetch(texvecino4GPU, vecinopxpypz);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = tex1D(texDelta, fabs(r));
  dlxp = tex1D(texDelta, fabs(rp));
  dlxm = tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = tex1D(texDelta, fabs(r));
  dlyp = tex1D(texDelta, fabs(rp));
  dlym = tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = tex1D(texDelta, fabs(r));
  dlzp = tex1D(texDelta, fabs(rp));
  dlzm = tex1D(texDelta, fabs(rm));

  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz);

  rho = dlxm * dlym * dlzm +
    dlxm * dlym * dlz  +
    dlxm * dlym * dlzp +
    dlxm * dly  * dlzm + 
    dlxm * dly  * dlz  +
    dlxm * dly  * dlzp +
    dlxm * dlyp * dlzm +
    dlxm * dlyp * dlz  +
    dlxm * dlyp * dlzp +
    dlx  * dlym * dlzm +
    dlx  * dlym * dlz  +
    dlx  * dlym * dlzp + 
    dlx  * dly  * dlzm + 
    dlx  * dly  * dlz  + 
    dlx  * dly  * dlzp + 
    dlx  * dlyp * dlzm + 
    dlx  * dlyp * dlz  + 
    dlx  * dlyp * dlzp + 
    dlxp * dlym * dlzm +
    dlxp * dlym * dlz  + 
    dlxp * dlym * dlzp +
    dlxp * dly  * dlzm + 
    dlxp * dly  * dlz  + 
    dlxp * dly  * dlzp +
    dlxp * dlyp * dlzm +
    dlxp * dlyp * dlz  +
    dlxp * dlyp * dlzp ;

  
  uold = vyboundaryGPU[nboundaryGPU+i];
  fact = 1.5 * volumeParticleGPU * volumeGPU * densfluidGPU * rho;
  u = uold + (fact * (v-uold) + fy*dtGPU) / (massParticleGPU + fact);
  vyboundaryGPU[nboundaryGPU+i] = u;

  f = (v - u) * 1.5 * volumeParticleGPU;

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;


  //FORCE IN THE Z DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icelz);
  vecino1 = tex1Dfetch(texvecino1GPU, icelz);
  vecino2 = tex1Dfetch(texvecino2GPU, icelz);
  vecino3 = tex1Dfetch(texvecino3GPU, icelz);
  vecino4 = tex1Dfetch(texvecino4GPU, icelz);
  vecino5 = tex1Dfetch(texvecino5GPU, icelz);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelz);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelz);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelz);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelz);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelz);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelz);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelz);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelz);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelz);  
  //DEFINE MORE NEIGHBORS
  int vecinopzmxmypz = tex1Dfetch(texvecino5GPU, vecinomxmypz);
  int vecinopzmxpz   = tex1Dfetch(texvecino5GPU, vecinomxpz);
  int vecinopzmxpypz = tex1Dfetch(texvecino5GPU, vecinomxpypz);
  int vecinopzmypz   = tex1Dfetch(texvecino5GPU, vecinomypz);
  int vecinopzpz     = tex1Dfetch(texvecino5GPU, vecino5);
  int vecinopzpypz   = tex1Dfetch(texvecino5GPU, vecinopypz);
  int vecinopzpxmypz = tex1Dfetch(texvecino5GPU, vecinopxmypz);
  int vecinopzpxpz   = tex1Dfetch(texvecino5GPU, vecinopxpz);
  int vecinopzpxpypz = tex1Dfetch(texvecino5GPU, vecinopxpypz);

  r =  (rx - rxcellGPU[icelz]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = tex1D(texDelta, fabs(r));
  dlxp = tex1D(texDelta, fabs(rp));
  dlxm = tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = tex1D(texDelta, fabs(r));
  dlyp = tex1D(texDelta, fabs(rp));
  dlym = tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = tex1D(texDelta, fabs(r));
  dlzp = tex1D(texDelta, fabs(rp));
  dlzm = tex1D(texDelta, fabs(rm));



  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);

  rho = dlxm * dlym * dlzm +
    dlxm * dlym * dlz  +
    dlxm * dlym * dlzp +
    dlxm * dly  * dlzm + 
    dlxm * dly  * dlz  +
    dlxm * dly  * dlzp +
    dlxm * dlyp * dlzm +
    dlxm * dlyp * dlz  +
    dlxm * dlyp * dlzp +
    dlx  * dlym * dlzm +
    dlx  * dlym * dlz  +
    dlx  * dlym * dlzp + 
    dlx  * dly  * dlzm + 
    dlx  * dly  * dlz  + 
    dlx  * dly  * dlzp + 
    dlx  * dlyp * dlzm + 
    dlx  * dlyp * dlz  + 
    dlx  * dlyp * dlzp + 
    dlxp * dlym * dlzm +
    dlxp * dlym * dlz  + 
    dlxp * dlym * dlzp +
    dlxp * dly  * dlzm + 
    dlxp * dly  * dlz  + 
    dlxp * dly  * dlzp +
    dlxp * dlyp * dlzm +
    dlxp * dlyp * dlz  +
    dlxp * dlyp * dlzp ;


  uold = vzboundaryGPU[nboundaryGPU+i];
  fact = 1.5 * volumeParticleGPU * volumeGPU * densfluidGPU * rho;
  u = uold + (fact * (v-uold) + fz*dtGPU) / (massParticleGPU + fact);
  vzboundaryGPU[nboundaryGPU+i] = u;

  f = (v - u) * 1.5 * volumeParticleGPU;

  offset = nboundaryGPU;
  fzboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;


  
}

































/*__global__ void nonBondedForceIncompressible2(double* rxcellGPU, 
			       double* rycellGPU, 
			       double* rzcellGPU,
			       double* vxboundaryGPU, 
			       double* vyboundaryGPU, 
			       double* vzboundaryGPU,
			       double* densityGPU,
			       double* fxboundaryGPU, 
			       double* fyboundaryGPU, 
			       double* fzboundaryGPU,
			       particlesincell* pc, 
			       int* errorKernel,
			       double* saveForceX, 
			       double* saveForceY, 
			       double* saveForceZ){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double fx = 0.;
  double fy = 0.;
  double fz = 0.;
  double f;
  
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

  double rxij, ryij, rzij, r2;

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;
  
  int icel;
  double r, rp, rm;

  {
    //int mxmy = mxNeighborsGPU * myNeighborsGPU;
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
    //printf("FFFFFF %i\n",jz);
  }
  
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellNonBonded,icel);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+icel);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }  
  //Particles in Cell vecino0
  vecino0 = tex1Dfetch(texneighbor0GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino0);
  //printf("RRRRRRRRR %i %i %i\n",i,icel,vecino0);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino0);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino1
  vecino1 = tex1Dfetch(texneighbor1GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino1);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino1);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
   }
  //Particles in Cell vecino2
  vecino2 = tex1Dfetch(texneighbor2GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino2);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino2);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino3
  vecino3 = tex1Dfetch(texneighbor3GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino3);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino3);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino4
  vecino4 = tex1Dfetch(texneighbor4GPU, icel);
  //printf("VECINO %i %i \n",icel,vecino4);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino4);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino4);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecino5
  vecino5 = tex1Dfetch(texneighbor5GPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino5);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino5);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpy
  vecinopxpy = tex1Dfetch(texneighborpxpyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpy);    
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmy
  vecinopxmy = tex1Dfetch(texneighborpxmyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmy);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpz
  vecinopxpz = tex1Dfetch(texneighborpxpzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmz
  vecinopxmz = tex1Dfetch(texneighborpxmzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpy
  vecinomxpy = tex1Dfetch(texneighbormxpyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpy);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij; 
  }
  //Particles in Cell vecinomxmy
  vecinomxmy = tex1Dfetch(texneighbormxmyGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmy);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmy);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpz
  vecinomxpz = tex1Dfetch(texneighbormxpzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxmz
  vecinomxmz = tex1Dfetch(texneighbormxmzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopypz
  vecinopypz = tex1Dfetch(texneighborpypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopymz
  vecinopymz = tex1Dfetch(texneighborpymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
   //Particles in Cell vecinomypz
  vecinomypz = tex1Dfetch(texneighbormypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomymz
  vecinomymz = tex1Dfetch(texneighbormymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpypz
  vecinopxpypz = tex1Dfetch(texneighborpxpypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxpymz
  vecinopxpymz = tex1Dfetch(texneighborpxpymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmypz
  vecinopxmypz = tex1Dfetch(texneighborpxmypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinopxmymz
  vecinopxmymz = tex1Dfetch(texneighborpxmymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpypz
  vecinomxpypz = tex1Dfetch(texneighbormxpypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxpymz
  vecinomxpymz = tex1Dfetch(texneighbormxpymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxmypz
  vecinomxmypz = tex1Dfetch(texneighbormxmypzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmypz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmypz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  //Particles in Cell vecinomxmymz
  vecinomxmymz = tex1Dfetch(texneighbormxmymzGPU, icel);
  np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmymz);
  for(int j=0;j<np;j++){
    int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmymz);
    rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
    rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
    ryij =  (ry - fetch_double(texryboundaryGPU,particle));
    ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
    rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
    rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
    r2 = rxij*rxij + ryij*ryij + rzij*rzij;
    f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
    fx += f * rxij;
    fy += f * ryij;
    fz += f * rzij;
  }
  
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * myGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }

  np = atomicAdd(&pc->countparticlesincellX[icelx],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellsGPU*np+icelx] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellY[icely],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[2]=np;
    return;
  }
  pc->partincellY[ncellsGPU*np+icely] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellZ[icelz],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[3]=np;
    return;
  }
  pc->partincellZ[ncellsGPU*np+icelz] = nboundaryGPU+i;

  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;

  //FORCE IN THE X DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icelx);
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecino5 = tex1Dfetch(texvecino5GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelx);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelx);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelx);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelx);
  int vecinopxpxpypz = tex1Dfetch(texvecino3GPU, vecinopxpypz);
  int vecinopxpxpymz = tex1Dfetch(texvecino3GPU, vecinopxpymz);
  int vecinopxpxmypz = tex1Dfetch(texvecino3GPU, vecinopxmypz);
  int vecinopxpxmymz = tex1Dfetch(texvecino3GPU, vecinopxmymz);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  int vecinopxpxpz   = tex1Dfetch(texvecino3GPU, vecinopxpz);
  int vecinopxpxmz   = tex1Dfetch(texvecino3GPU, vecinopxmz);

  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(r);//tex1D(texDelta, fabs(r));
  dlxp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlxm = delta(rm);//tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(r);//tex1D(texDelta, fabs(r));
  dlyp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlym = delta(rm);//tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(r);//tex1D(texDelta, fabs(r));
  dlzp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlzm = delta(rm);//tex1D(texDelta, fabs(rm));

  double v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);

  

  double rho = dlxm * dlym * dlzm +
    dlxm * dlym * dlz  +
    dlxm * dlym * dlzp +
    dlxm * dly  * dlzm + 
    dlxm * dly  * dlz  +
    dlxm * dly  * dlzp +
    dlxm * dlyp * dlzm +
    dlxm * dlyp * dlz  +
    dlxm * dlyp * dlzp +
    dlx  * dlym * dlzm +
    dlx  * dlym * dlz  +
    dlx  * dlym * dlzp + 
    dlx  * dly  * dlzm + 
    dlx  * dly  * dlz  + 
    dlx  * dly  * dlzp + 
    dlx  * dlyp * dlzm + 
    dlx  * dlyp * dlz  + 
    dlx  * dlyp * dlzp + 
    dlxp * dlym * dlzm +
    dlxp * dlym * dlz  + 
    dlxp * dlym * dlzp +
    dlxp * dly  * dlzm + 
    dlxp * dly  * dlz  + 
    dlxp * dly  * dlzp +
    dlxp * dlyp * dlzm +
    dlxp * dlyp * dlz  +
    dlxp * dlyp * dlzp ;

  
  double u, uold, fact;
  
  uold = vxboundaryGPU[nboundaryGPU+i];
  fact = 1.5 * volumeParticleGPU * volumeGPU * 1;
  //fact = volumeParticleGPU * volumeGPU * densfluidGPU;
  u = uold + (fact * (v-uold) + fx*dtGPU)/(massParticleGPU + fact);
  vxboundaryGPU[nboundaryGPU+i] = u;


  f = 1.5 * (v - u) * volumeParticleGPU;

  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;
  
  
  //FORCE IN THE Y DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icely);
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecino5 = tex1Dfetch(texvecino5GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icely);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icely);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icely);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icely);  
  //DEFINE MORE NEIGHBORS
  int vecinopymxpymz = tex1Dfetch(texvecino4GPU, vecinomxpymz);
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopymxpypz = tex1Dfetch(texvecino4GPU, vecinomxpypz);
  int vecinopypymz   = tex1Dfetch(texvecino4GPU, vecinopymz);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypypz   = tex1Dfetch(texvecino4GPU, vecinopypz);
  int vecinopypxpymz = tex1Dfetch(texvecino4GPU, vecinopxpymz);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);
  int vecinopypxpypz = tex1Dfetch(texvecino4GPU, vecinopxpypz);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(r);//tex1D(texDelta, fabs(r));
  dlxp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlxm = delta(rm);//tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(r);//tex1D(texDelta, fabs(r));
  dlyp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlym = delta(rm);//tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(r);//tex1D(texDelta, fabs(r));
  dlzp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlzm = delta(rm);//tex1D(texDelta, fabs(rm));


  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz);

  rho = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomxmz]) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecino2]) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinomxpz]) +
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecinomxpymz]) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[vecinomxpy]) +
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecinomxpypz]) +
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinopymxpymz]) +
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecinopymxpy]) +
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopymxpypz]) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecino0]) +
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[icely]) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecino5]) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[vecinopymz]) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icely]        + densityGPU[vecino4]) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopypz]) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecinopypymz]) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopypy]) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopypypz]) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxmz]) +
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecino3]) +
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopxpz]) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecinopxpymz]) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpy]) + 
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz]   + densityGPU[vecinopxpypz]) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopypxpymz]) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy]   + densityGPU[vecinopypxpy]) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopypxpypz]);

  uold = vyboundaryGPU[nboundaryGPU+i];
  //fact = volumeParticleGPU * volumeGPU * rho;
  fact = 1.5 * volumeParticleGPU * volumeGPU * densfluidGPU;
  u = uold + (fact * (v-uold) + fy * dtGPU)/(massParticleGPU + fact);
  vyboundaryGPU[nboundaryGPU+i] = u;

  f = (v - u) * volumeParticleGPU * 1.5 ;

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;
  
  
  //FORCE IN THE Z DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icelz);
  vecino1 = tex1Dfetch(texvecino1GPU, icelz);
  vecino2 = tex1Dfetch(texvecino2GPU, icelz);
  vecino3 = tex1Dfetch(texvecino3GPU, icelz);
  vecino4 = tex1Dfetch(texvecino4GPU, icelz);
  vecino5 = tex1Dfetch(texvecino5GPU, icelz);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelz);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelz);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelz);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelz);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelz);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelz);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelz);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelz);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelz);  
  //DEFINE MORE NEIGHBORS
  int vecinopzmxmypz = tex1Dfetch(texvecino5GPU, vecinomxmypz);
  int vecinopzmxpz   = tex1Dfetch(texvecino5GPU, vecinomxpz);
  int vecinopzmxpypz = tex1Dfetch(texvecino5GPU, vecinomxpypz);
  int vecinopzmypz   = tex1Dfetch(texvecino5GPU, vecinomypz);
  int vecinopzpz     = tex1Dfetch(texvecino5GPU, vecino5);
  int vecinopzpypz   = tex1Dfetch(texvecino5GPU, vecinopypz);
  int vecinopzpxmypz = tex1Dfetch(texvecino5GPU, vecinopxmypz);
  int vecinopzpxpz   = tex1Dfetch(texvecino5GPU, vecinopxpz);
  int vecinopzpxpypz = tex1Dfetch(texvecino5GPU, vecinopxpypz);

  r =  (rx - rxcellGPU[icelz]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(r);//tex1D(texDelta, fabs(r));
  dlxp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlxm = delta(rm);//tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(r);//tex1D(texDelta, fabs(r));
  dlyp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlym = delta(rm);//tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(r);//tex1D(texDelta, fabs(r));
  dlzp = delta(rp);//tex1D(texDelta, fabs(rp));
  dlzm = delta(rm);//tex1D(texDelta, fabs(rm));


  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);

  rho = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomxmy]) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecinomxmypz]) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinopzmxmypz]) + 
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecino2]) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[vecinomxpz]) + 
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecinopzmxpz]) + 
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinomxpy]) + 
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecinomxpypz]) + 
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopzmxpypz]) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecino1]) + 
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[vecinomypz]) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecinopzmypz]) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[icelz]) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icelz]        + densityGPU[vecino5]) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopzpz]) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecino4]) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopypz]) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopzpypz]) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxmy]) +
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecinopxmypz]) +
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopzpxmypz]) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecino3]) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpz]) + 
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz]   + densityGPU[vecinopzpxpz]) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopxpy]) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy]   + densityGPU[vecinopxpypz]) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopzpxpypz]);

  uold = vzboundaryGPU[nboundaryGPU+i];
  //fact = volumeParticleGPU * volumeGPU * rho;
  fact = 1.5 * volumeParticleGPU * volumeGPU * densfluidGPU;
  u = uold + (fact * (v-uold) + fz * dtGPU)/(massParticleGPU + fact);
  vzboundaryGPU[nboundaryGPU+i] = u;

  f = (v - u) * volumeParticleGPU * 1.5;

  offset = nboundaryGPU;
  fzboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;

  
}
*/
