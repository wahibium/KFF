// Filename: kernelSpreadParticles.cu
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


__global__ void kernelSpreadParticles(double* rxcellGPU, 
				      double* rycellGPU, 
				      double* rzcellGPU,
				      double* vxboundaryGPU,
				      double* vyboundaryGPU,
				      double* vzboundaryGPU,
				      double* fxboundaryGPU, 
				      double* fyboundaryGPU, 
				      double* fzboundaryGPU,
				      particlesincell* pc, 
				      int* errorKernel){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double fx = 0.;//0.188495559215;
  double fy = 0.;
  double fz = 0.;
  double f;
 
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  //fx = -sin(rx)*cos(rx) - 2*sin(rx)*cos(rx)*sin(ry)  + 0.35;
  //fy = -sin(ry)*cos(ry) - cos(ry)*sin(rx)*sin(rx) + sin(ry)*cos(ry)*cos(ry);
  //fy = -ry - rx*rx + ry*ry;
  //fx = -0.01*rx;
  //fy = -0.01*ry;
  //fz = -0.01*rz;

  double ux = vxboundaryGPU[nboundaryGPU+i];
  double uy = vyboundaryGPU[nboundaryGPU+i];
  double uz = vzboundaryGPU[nboundaryGPU+i];
  
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

  double dlxS, dlxpS, dlxmS;
  double dlyS, dlypS, dlymS;
  double dlzS, dlzpS, dlzmS;

  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx  = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  dlxS = functionDeltaDerived(1.5*r);
  dlxpS = functionDeltaDerived(1.5*rp);
  dlxmS = functionDeltaDerived(1.5*rm);
                              
  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  dlyS = functionDeltaDerived(1.5*r);
  dlypS = functionDeltaDerived(1.5*rp);
  dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  dlzS = functionDeltaDerived(1.5*r);
  dlzpS = functionDeltaDerived(1.5*rp);
  dlzmS = functionDeltaDerived(1.5*rm);

    
  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i]        = -dlxp * dlyp * dlzp * fx +
    (dlxpS * dlyp  * dlzp  * ux * ux +
     dlxp  * dlypS * dlzp  * ux * uy +
     dlxp  * dlyp  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fx + 
    (dlxpS * dlyp  * dlz  * ux * ux +
     dlxp  * dlypS * dlz  * ux * uy +
     dlxp  * dlyp  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fx + 
    (dlxpS * dlyp  * dlzm  * ux * ux +
     dlxp  * dlypS * dlzm  * ux * uy +
     dlxp  * dlyp  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fx + 
    (dlxpS * dly  * dlzp  * ux * ux +
     dlxp  * dlyS * dlzp  * ux * uy +
     dlxp  * dly  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fx +
    (dlxpS * dly  * dlz  * ux * ux +
     dlxp  * dlyS * dlz  * ux * uy +
     dlxp  * dly  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fx + 
    (dlxpS * dly  * dlzm  * ux * ux +
     dlxp  * dlyS * dlzm  * ux * uy +
     dlxp  * dly  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fx +
    (dlxpS * dlym  * dlzp  * ux * ux +
     dlxp  * dlymS * dlzp  * ux * uy +
     dlxp  * dlym  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fx + 
    (dlxpS * dlym  * dlz  * ux * ux +
     dlxp  * dlymS * dlz  * ux * uy +
     dlxp  * dlym  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fx +
    (dlxpS * dlym  * dlzm  * ux * ux +
     dlxp  * dlymS * dlzm  * ux * uy +
     dlxp  * dlym  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fx +
    (dlxS * dlyp  * dlzp  * ux * ux +
     dlx  * dlypS * dlzp  * ux * uy +
     dlx  * dlyp  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fx +
    (dlxS * dlyp  * dlz  * ux * ux +
     dlx  * dlypS * dlz  * ux * uy +
     dlx  * dlyp  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fx +
    (dlxS * dlyp  * dlzm  * ux * ux +
     dlx  * dlypS * dlzm  * ux * uy +
     dlx  * dlyp  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fx +
    (dlxS * dly  * dlzp  * ux * ux +
     dlx  * dlyS * dlzp  * ux * uy +
     dlx  * dly  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fx +
    (dlxS * dly  * dlz  * ux * ux +
     dlx  * dlyS * dlz  * ux * uy +
     dlx  * dly  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fx + 
    (dlxS * dly  * dlzm  * ux * ux +
     dlx  * dlyS * dlzm  * ux * uy +
     dlx  * dly  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fx +
    (dlxS * dlym  * dlzp  * ux * ux +
     dlx  * dlymS * dlzp  * ux * uy +
     dlx  * dlym  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fx +
    (dlxS * dlym  * dlz  * ux * ux +
     dlx  * dlymS * dlz  * ux * uy +
     dlx  * dlym  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fx +
    (dlxS * dlym  * dlzm  * ux * ux +
     dlx  * dlymS * dlzm  * ux * uy +
     dlx  * dlym  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fx + 
    (dlxmS * dlyp  * dlzp  * ux * ux +
     dlxm  * dlypS * dlzp  * ux * uy +
     dlxm  * dlyp  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fx + 
    (dlxmS * dlyp  * dlz  * ux * ux +
     dlxm  * dlypS * dlz  * ux * uy +
     dlxm  * dlyp  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fx +
    (dlxmS * dlyp  * dlzm  * ux * ux +
     dlxm  * dlypS * dlzm  * ux * uy +
     dlxm  * dlyp  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fx + 
    (dlxmS * dly  * dlzp  * ux * ux +
     dlxm  * dlyS * dlzp  * ux * uy +
     dlxm  * dly  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fx + 
    (dlxmS * dly  * dlz  * ux * ux +
     dlxm  * dlyS * dlz  * ux * uy +
     dlxm  * dly  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fx + 
    (dlxmS * dly  * dlzm  * ux * ux +
     dlxm  * dlyS * dlzm  * ux * uy +
     dlxm  * dly  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fx + 
    (dlxmS * dlym  * dlzp  * ux * ux +
     dlxm  * dlymS * dlzp  * ux * uy +
     dlxm  * dlym  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fx + 
    (dlxmS * dlym  * dlz  * ux * ux +
     dlxm  * dlymS * dlz  * ux * uy +
     dlxm  * dlym  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fx + 
    (dlxmS * dlym  * dlzm  * ux * ux +
     dlxm  * dlymS * dlzm  * ux * uy +
     dlxm  * dlym  * dlzmS * ux * uz) * massParticleGPU;
  
  
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
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  dlxS = functionDeltaDerived(1.5*r);
  dlxpS = functionDeltaDerived(1.5*rp);
  dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  dlyS = functionDeltaDerived(1.5*r);
  dlypS = functionDeltaDerived(1.5*rp);
  dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  dlzS = functionDeltaDerived(1.5*r);
  dlzpS = functionDeltaDerived(1.5*rp);
  dlzmS = functionDeltaDerived(1.5*rm);

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fy +
    (dlxpS * dlyp  * dlzp  * uy * ux +
     dlxp  * dlypS * dlzp  * uy * uy +
     dlxp  * dlyp  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fy + 
    (dlxpS * dlyp  * dlz  * uy * ux +
     dlxp  * dlypS * dlz  * uy * uy +
     dlxp  * dlyp  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fy + 
    (dlxpS * dlyp  * dlzm  * uy * ux +
     dlxp  * dlypS * dlzm  * uy * uy +
     dlxp  * dlyp  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fy + 
    (dlxpS * dly  * dlzp  * uy * ux +
     dlxp  * dlyS * dlzp  * uy * uy +
     dlxp  * dly  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fy +
    (dlxpS * dly  * dlz  * uy * ux +
     dlxp  * dlyS * dlz  * uy * uy +
     dlxp  * dly  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fy + 
    (dlxpS * dly  * dlzm  * uy * ux +
     dlxp  * dlyS * dlzm  * uy * uy +
     dlxp  * dly  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fy +
    (dlxpS * dlym  * dlzp  * uy * ux +
     dlxp  * dlymS * dlzp  * uy * uy +
     dlxp  * dlym  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fy + 
    (dlxpS * dlym  * dlz  * uy * ux +
     dlxp  * dlymS * dlz  * uy * uy +
     dlxp  * dlym  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fy +
    (dlxpS * dlym  * dlzm  * uy * ux +
     dlxp  * dlymS * dlzm  * uy * uy +
     dlxp  * dlym  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fy +
    (dlxS * dlyp  * dlzp  * uy * ux +
     dlx  * dlypS * dlzp  * uy * uy +
     dlx  * dlyp  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fy +
    (dlxS * dlyp  * dlz  * uy * ux +
     dlx  * dlypS * dlz  * uy * uy +
     dlx  * dlyp  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fy +
    (dlxS * dlyp  * dlzm  * uy * ux +
     dlx  * dlypS * dlzm  * uy * uy +
     dlx  * dlyp  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fy +
    (dlxS * dly  * dlzp  * uy * ux +
     dlx  * dlyS * dlzp  * uy * uy +
     dlx  * dly  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fy +
    (dlxS * dly  * dlz  * uy * ux +
     dlx  * dlyS * dlz  * uy * uy +
     dlx  * dly  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fy + 
    (dlxS * dly  * dlzm  * uy * ux +
     dlx  * dlyS * dlzm  * uy * uy +
     dlx  * dly  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fy +
    (dlxS * dlym  * dlzp  * uy * ux +
     dlx  * dlymS * dlzp  * uy * uy +
     dlx  * dlym  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fy +
    (dlxS * dlym  * dlz  * uy * ux +
     dlx  * dlymS * dlz  * uy * uy +
     dlx  * dlym  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fy +
    (dlxS * dlym  * dlzm  * uy * ux +
     dlx  * dlymS * dlzm  * uy * uy +
     dlx  * dlym  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fy + 
    (dlxmS * dlyp  * dlzp  * uy * ux +
     dlxm  * dlypS * dlzp  * uy * uy +
     dlxm  * dlyp  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fy + 
    (dlxmS * dlyp  * dlz  * uy * ux +
     dlxm  * dlypS * dlz  * uy * uy +
     dlxm  * dlyp  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fy +
    (dlxmS * dlyp  * dlzm  * uy * ux +
     dlxm  * dlypS * dlzm  * uy * uy +
     dlxm  * dlyp  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fy + 
    (dlxmS * dly  * dlzp  * uy * ux +
     dlxm  * dlyS * dlzp  * uy * uy +
     dlxm  * dly  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fy + 
    (dlxmS * dly  * dlz  * uy * ux +
     dlxm  * dlyS * dlz  * uy * uy +
     dlxm  * dly  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fy + 
    (dlxmS * dly  * dlzm  * uy * ux +
     dlxm  * dlyS * dlzm  * uy * uy +
     dlxm  * dly  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fy + 
    (dlxmS * dlym  * dlzp  * uy * ux +
     dlxm  * dlymS * dlzp  * uy * uy +
     dlxm  * dlym  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fy + 
    (dlxmS * dlym  * dlz  * uy * ux +
     dlxm  * dlymS * dlz  * uy * uy +
     dlxm  * dlym  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fy + 
    (dlxmS * dlym  * dlzm  * uy * ux +
     dlxm  * dlymS * dlzm  * uy * uy +
     dlxm  * dlym  * dlzmS * uy * uz) * massParticleGPU;


  
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
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  dlxS = functionDeltaDerived(1.5*r);
  dlxpS = functionDeltaDerived(1.5*rp);
  dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  dlyS = functionDeltaDerived(1.5*r);
  dlypS = functionDeltaDerived(1.5*rp);
  dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  dlzS = functionDeltaDerived(1.5*r);
  dlzpS = functionDeltaDerived(1.5*rp);
  dlzmS = functionDeltaDerived(1.5*rm);


  offset = nboundaryGPU;
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fz +
    (dlxpS * dlyp  * dlzp  * uz * ux +
     dlxp  * dlypS * dlzp  * uz * uy +
     dlxp  * dlyp  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fz + 
    (dlxpS * dlyp  * dlz  * uz * ux +
     dlxp  * dlypS * dlz  * uz * uy +
     dlxp  * dlyp  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fz + 
    (dlxpS * dlyp  * dlzm  * uz * ux +
     dlxp  * dlypS * dlzm  * uz * uy +
     dlxp  * dlyp  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fz + 
    (dlxpS * dly  * dlzp  * uz * ux +
     dlxp  * dlyS * dlzp  * uz * uy +
     dlxp  * dly  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fz +
    (dlxpS * dly  * dlz  * uz * ux +
     dlxp  * dlyS * dlz  * uz * uy +
     dlxp  * dly  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fz + 
    (dlxpS * dly  * dlzm  * uz * ux +
     dlxp  * dlyS * dlzm  * uz * uy +
     dlxp  * dly  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fz +
    (dlxpS * dlym  * dlzp  * uz * ux +
     dlxp  * dlymS * dlzp  * uz * uy +
     dlxp  * dlym  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fz + 
    (dlxpS * dlym  * dlz  * uz * ux +
     dlxp  * dlymS * dlz  * uz * uy +
     dlxp  * dlym  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fz +
    (dlxpS * dlym  * dlzm  * uz * ux +
     dlxp  * dlymS * dlzm  * uz * uy +
     dlxp  * dlym  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fz +
    (dlxS * dlyp  * dlzp  * uz * ux +
     dlx  * dlypS * dlzp  * uz * uy +
     dlx  * dlyp  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fz +
    (dlxS * dlyp  * dlz  * uz * ux +
     dlx  * dlypS * dlz  * uz * uy +
     dlx  * dlyp  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fz +
    (dlxS * dlyp  * dlzm  * uz * ux +
     dlx  * dlypS * dlzm  * uz * uy +
     dlx  * dlyp  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fz +
    (dlxS * dly  * dlzp  * uz * ux +
     dlx  * dlyS * dlzp  * uz * uy +
     dlx  * dly  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fz +
    (dlxS * dly  * dlz  * uz * ux +
     dlx  * dlyS * dlz  * uz * uy +
     dlx  * dly  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fz + 
    (dlxS * dly  * dlzm  * uz * ux +
     dlx  * dlyS * dlzm  * uz * uy +
     dlx  * dly  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fz +
    (dlxS * dlym  * dlzp  * uz * ux +
     dlx  * dlymS * dlzp  * uz * uy +
     dlx  * dlym  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fz +
    (dlxS * dlym  * dlz  * uz * ux +
     dlx  * dlymS * dlz  * uz * uy +
     dlx  * dlym  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fz +
    (dlxS * dlym  * dlzm  * uz * ux +
     dlx  * dlymS * dlzm  * uz * uy +
     dlx  * dlym  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fz + 
    (dlxmS * dlyp  * dlzp  * uz * ux +
     dlxm  * dlypS * dlzp  * uz * uy +
     dlxm  * dlyp  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fz + 
    (dlxmS * dlyp  * dlz  * uz * ux +
     dlxm  * dlypS * dlz  * uz * uy +
     dlxm  * dlyp  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fz +
    (dlxmS * dlyp  * dlzm  * uz * ux +
     dlxm  * dlypS * dlzm  * uz * uy +
     dlxm  * dlyp  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fz + 
    (dlxmS * dly  * dlzp  * uz * ux +
     dlxm  * dlyS * dlzp  * uz * uy +
     dlxm  * dly  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fz + 
    (dlxmS * dly  * dlz  * uz * ux +
     dlxm  * dlyS * dlz  * uz * uy +
     dlxm  * dly  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fz + 
    (dlxmS * dly  * dlzm  * uz * ux +
     dlxm  * dlyS * dlzm  * uz * uy +
     dlxm  * dly  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fz + 
    (dlxmS * dlym  * dlzp  * uz * ux +
     dlxm  * dlymS * dlzp  * uz * uy +
     dlxm  * dlym  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fz + 
    (dlxmS * dlym  * dlz  * uz * ux +
     dlxm  * dlymS * dlz  * uz * uy +
     dlxm  * dlym  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fz + 
    (dlxmS * dlym  * dlzm  * uz * ux +
     dlxm  * dlymS * dlzm  * uz * uy +
     dlxm  * dlym  * dlzmS * uz * uz) * massParticleGPU;

  
}
















__global__ void kernelSpreadParticlesAdvection(double* rxcellGPU, 
					       double* rycellGPU, 
					       double* rzcellGPU,
					       double* vxboundaryGPU,
					       double* vyboundaryGPU,
					       double* vzboundaryGPU,
					       double* fxboundaryGPU, 
					       double* fyboundaryGPU, 
					       double* fzboundaryGPU,
					       particlesincell* pc, 
					       int* errorKernel){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  double ux = vxboundaryGPU[nboundaryGPU+i];
  double uy = vyboundaryGPU[nboundaryGPU+i];
  double uz = vzboundaryGPU[nboundaryGPU+i];

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopypz, vecinopymz, vecinomypz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz;
  
  double r, rp, rm;
  
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

  int np = atomicAdd(&pc->countparticlesincellX[icelx],1);
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
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  int vecinopxpxpypz = tex1Dfetch(texvecino3GPU, vecinopxpypz);
  int vecinopxpxpymz = tex1Dfetch(texvecino3GPU, vecinopxpymz);
  int vecinopxpxmypz = tex1Dfetch(texvecino3GPU, vecinopxmypz);
  int vecinopxpxmymz = tex1Dfetch(texvecino3GPU, vecinopxmymz);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  int vecinopxpxpz   = tex1Dfetch(texvecino3GPU, vecinopxpz);
  int vecinopxpxmz   = tex1Dfetch(texvecino3GPU, vecinopxmz);

  double dlxS, dlxpS, dlxmS;
  double dlyS, dlypS, dlymS;
  double dlzS, dlzpS, dlzmS;

  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  dlxS = functionDeltaDerived(1.5*r);
  dlxpS = functionDeltaDerived(1.5*rp);
  dlxmS = functionDeltaDerived(1.5*rm);
                              
  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  dlyS = functionDeltaDerived(1.5*r);
  dlypS = functionDeltaDerived(1.5*rp);
  dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  dlzS = functionDeltaDerived(1.5*r);
  dlzpS = functionDeltaDerived(1.5*rp);
  dlzmS = functionDeltaDerived(1.5*rm);

    
  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = (dlxpS * dlyp  * dlzp  * ux * ux +
			     dlxp  * dlypS * dlzp  * ux * uy +
			     dlxp  * dlyp  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = (dlxpS * dlyp  * dlz  * ux * ux +
			     dlxp  * dlypS * dlz  * ux * uy +
			     dlxp  * dlyp  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = (dlxpS * dlyp  * dlzm  * ux * ux +
			     dlxp  * dlypS * dlzm  * ux * uy +
			     dlxp  * dlyp  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = (dlxpS * dly  * dlzp  * ux * ux +
			     dlxp  * dlyS * dlzp  * ux * uy +
			     dlxp  * dly  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = (dlxpS * dly  * dlz  * ux * ux +
			     dlxp  * dlyS * dlz  * ux * uy +
			     dlxp  * dly  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = (dlxpS * dly  * dlzm  * ux * ux +
			     dlxp  * dlyS * dlzm  * ux * uy +
			     dlxp  * dly  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = (dlxpS * dlym  * dlzp  * ux * ux +
			     dlxp  * dlymS * dlzp  * ux * uy +
			     dlxp  * dlym  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = (dlxpS * dlym  * dlz  * ux * ux +
			     dlxp  * dlymS * dlz  * ux * uy +
			     dlxp  * dlym  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = (dlxpS * dlym  * dlzm  * ux * ux +
			     dlxp  * dlymS * dlzm  * ux * uy +
			     dlxp  * dlym  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = (dlxS * dlyp  * dlzp  * ux * ux +
			     dlx  * dlypS * dlzp  * ux * uy +
			     dlx  * dlyp  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = (dlxS * dlyp  * dlz  * ux * ux +
			     dlx  * dlypS * dlz  * ux * uy +
			     dlx  * dlyp  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = (dlxS * dlyp  * dlzm  * ux * ux +
			     dlx  * dlypS * dlzm  * ux * uy +
			     dlx  * dlyp  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = (dlxS * dly  * dlzp  * ux * ux +
			     dlx  * dlyS * dlzp  * ux * uy +
			     dlx  * dly  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = (dlxS * dly  * dlz  * ux * ux +
			     dlx  * dlyS * dlz  * ux * uy +
			     dlx  * dly  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = (dlxS * dly  * dlzm  * ux * ux +
			     dlx  * dlyS * dlzm  * ux * uy +
			     dlx  * dly  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = (dlxS * dlym  * dlzp  * ux * ux +
			     dlx  * dlymS * dlzp  * ux * uy +
			     dlx  * dlym  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = (dlxS * dlym  * dlz  * ux * ux +
			     dlx  * dlymS * dlz  * ux * uy +
			     dlx  * dlym  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = (dlxS * dlym  * dlzm  * ux * ux +
			     dlx  * dlymS * dlzm  * ux * uy +
			     dlx  * dlym  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = (dlxmS * dlyp  * dlzp  * ux * ux +
			     dlxm  * dlypS * dlzp  * ux * uy +
			     dlxm  * dlyp  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = (dlxmS * dlyp  * dlz  * ux * ux +
			     dlxm  * dlypS * dlz  * ux * uy +
			     dlxm  * dlyp  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = (dlxmS * dlyp  * dlzm  * ux * ux +
			     dlxm  * dlypS * dlzm  * ux * uy +
			     dlxm  * dlyp  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = (dlxmS * dly  * dlzp  * ux * ux +
			     dlxm  * dlyS * dlzp  * ux * uy +
			     dlxm  * dly  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = (dlxmS * dly  * dlz  * ux * ux +
			     dlxm  * dlyS * dlz  * ux * uy +
			     dlxm  * dly  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = (dlxmS * dly  * dlzm  * ux * ux +
			     dlxm  * dlyS * dlzm  * ux * uy +
			     dlxm  * dly  * dlzmS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = (dlxmS * dlym  * dlzp  * ux * ux +
			     dlxm  * dlymS * dlzp  * ux * uy +
			     dlxm  * dlym  * dlzpS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = (dlxmS * dlym  * dlz  * ux * ux +
			     dlxm  * dlymS * dlz  * ux * uy +
			     dlxm  * dlym  * dlzS * ux * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = (dlxmS * dlym  * dlzm  * ux * ux +
			     dlxm  * dlymS * dlzm  * ux * uy +
			     dlxm  * dlym  * dlzmS * ux * uz) * massParticleGPU;
  
  
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
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
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
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  dlxS = functionDeltaDerived(1.5*r);
  dlxpS = functionDeltaDerived(1.5*rp);
  dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  dlyS = functionDeltaDerived(1.5*r);
  dlypS = functionDeltaDerived(1.5*rp);
  dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  dlzS = functionDeltaDerived(1.5*r);
  dlzpS = functionDeltaDerived(1.5*rp);
  dlzmS = functionDeltaDerived(1.5*rm);

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = (dlxpS * dlyp  * dlzp  * uy * ux +
			     dlxp  * dlypS * dlzp  * uy * uy +
			     dlxp  * dlyp  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = (dlxpS * dlyp  * dlz  * uy * ux +
			     dlxp  * dlypS * dlz  * uy * uy +
			     dlxp  * dlyp  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = (dlxpS * dlyp  * dlzm  * uy * ux +
			     dlxp  * dlypS * dlzm  * uy * uy +
			     dlxp  * dlyp  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = (dlxpS * dly  * dlzp  * uy * ux +
			     dlxp  * dlyS * dlzp  * uy * uy +
			     dlxp  * dly  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = (dlxpS * dly  * dlz  * uy * ux +
			     dlxp  * dlyS * dlz  * uy * uy +
			     dlxp  * dly  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = (dlxpS * dly  * dlzm  * uy * ux +
			     dlxp  * dlyS * dlzm  * uy * uy +
			     dlxp  * dly  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = (dlxpS * dlym  * dlzp  * uy * ux +
			     dlxp  * dlymS * dlzp  * uy * uy +
			     dlxp  * dlym  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = (dlxpS * dlym  * dlz  * uy * ux +
			     dlxp  * dlymS * dlz  * uy * uy +
			     dlxp  * dlym  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = (dlxpS * dlym  * dlzm  * uy * ux +
			     dlxp  * dlymS * dlzm  * uy * uy +
			     dlxp  * dlym  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = (dlxS * dlyp  * dlzp  * uy * ux +
			     dlx  * dlypS * dlzp  * uy * uy +
			     dlx  * dlyp  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = (dlxS * dlyp  * dlz  * uy * ux +
			     dlx  * dlypS * dlz  * uy * uy +
			     dlx  * dlyp  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = (dlxS * dlyp  * dlzm  * uy * ux +
			     dlx  * dlypS * dlzm  * uy * uy +
			     dlx  * dlyp  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = (dlxS * dly  * dlzp  * uy * ux +
			     dlx  * dlyS * dlzp  * uy * uy +
			     dlx  * dly  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = (dlxS * dly  * dlz  * uy * ux +
			     dlx  * dlyS * dlz  * uy * uy +
			     dlx  * dly  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = (dlxS * dly  * dlzm  * uy * ux +
			     dlx  * dlyS * dlzm  * uy * uy +
			     dlx  * dly  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = (dlxS * dlym  * dlzp  * uy * ux +
			     dlx  * dlymS * dlzp  * uy * uy +
			     dlx  * dlym  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = (dlxS * dlym  * dlz  * uy * ux +
			     dlx  * dlymS * dlz  * uy * uy +
			     dlx  * dlym  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = (dlxS * dlym  * dlzm  * uy * ux +
			     dlx  * dlymS * dlzm  * uy * uy +
			     dlx  * dlym  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = (dlxmS * dlyp  * dlzp  * uy * ux +
			     dlxm  * dlypS * dlzp  * uy * uy +
			     dlxm  * dlyp  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = (dlxmS * dlyp  * dlz  * uy * ux +
			     dlxm  * dlypS * dlz  * uy * uy +
			     dlxm  * dlyp  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = (dlxmS * dlyp  * dlzm  * uy * ux +
			     dlxm  * dlypS * dlzm  * uy * uy +
			     dlxm  * dlyp  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = (dlxmS * dly  * dlzp  * uy * ux +
			     dlxm  * dlyS * dlzp  * uy * uy +
			     dlxm  * dly  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = (dlxmS * dly  * dlz  * uy * ux +
			     dlxm  * dlyS * dlz  * uy * uy +
			     dlxm  * dly  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = (dlxmS * dly  * dlzm  * uy * ux +
			     dlxm  * dlyS * dlzm  * uy * uy +
			     dlxm  * dly  * dlzmS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = (dlxmS * dlym  * dlzp  * uy * ux +
			     dlxm  * dlymS * dlzp  * uy * uy +
			     dlxm  * dlym  * dlzpS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = (dlxmS * dlym  * dlz  * uy * ux +
			     dlxm  * dlymS * dlz  * uy * uy +
			     dlxm  * dlym  * dlzS * uy * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = (dlxmS * dlym  * dlzm  * uy * ux +
			     dlxm  * dlymS * dlzm  * uy * uy +
			     dlxm  * dlym  * dlzmS * uy * uz) * massParticleGPU;


  
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
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
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
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  dlxS = functionDeltaDerived(1.5*r);
  dlxpS = functionDeltaDerived(1.5*rp);
  dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  dlyS = functionDeltaDerived(1.5*r);
  dlypS = functionDeltaDerived(1.5*rp);
  dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  dlzS = functionDeltaDerived(1.5*r);
  dlzpS = functionDeltaDerived(1.5*rp);
  dlzmS = functionDeltaDerived(1.5*rm);


  offset = nboundaryGPU;
  fzboundaryGPU[offset+i] = (dlxpS * dlyp  * dlzp  * uz * ux +
			     dlxp  * dlypS * dlzp  * uz * uy +
			     dlxp  * dlyp  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = (dlxpS * dlyp  * dlz  * uz * ux +
			     dlxp  * dlypS * dlz  * uz * uy +
			     dlxp  * dlyp  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = (dlxpS * dlyp  * dlzm  * uz * ux +
			     dlxp  * dlypS * dlzm  * uz * uy +
			     dlxp  * dlyp  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = (dlxpS * dly  * dlzp  * uz * ux +
			     dlxp  * dlyS * dlzp  * uz * uy +
			     dlxp  * dly  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = (dlxpS * dly  * dlz  * uz * ux +
			     dlxp  * dlyS * dlz  * uz * uy +
			     dlxp  * dly  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = (dlxpS * dly  * dlzm  * uz * ux +
			     dlxp  * dlyS * dlzm  * uz * uy +
			     dlxp  * dly  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = (dlxpS * dlym  * dlzp  * uz * ux +
			     dlxp  * dlymS * dlzp  * uz * uy +
			     dlxp  * dlym  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = (dlxpS * dlym  * dlz  * uz * ux +
			     dlxp  * dlymS * dlz  * uz * uy +
			     dlxp  * dlym  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = (dlxpS * dlym  * dlzm  * uz * ux +
			     dlxp  * dlymS * dlzm  * uz * uy +
			     dlxp  * dlym  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = (dlxS * dlyp  * dlzp  * uz * ux +
			     dlx  * dlypS * dlzp  * uz * uy +
			     dlx  * dlyp  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] =  (dlxS * dlyp  * dlz  * uz * ux +
			      dlx  * dlypS * dlz  * uz * uy +
			      dlx  * dlyp  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = (dlxS * dlyp  * dlzm  * uz * ux +
			     dlx  * dlypS * dlzm  * uz * uy +
			     dlx  * dlyp  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = (dlxS * dly  * dlzp  * uz * ux +
			     dlx  * dlyS * dlzp  * uz * uy +
			     dlx  * dly  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = (dlxS * dly  * dlz  * uz * ux +
			     dlx  * dlyS * dlz  * uz * uy +
			     dlx  * dly  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = (dlxS * dly  * dlzm  * uz * ux +
			     dlx  * dlyS * dlzm  * uz * uy +
			     dlx  * dly  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = (dlxS * dlym  * dlzp  * uz * ux +
			     dlx  * dlymS * dlzp  * uz * uy +
			     dlx  * dlym  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = (dlxS * dlym  * dlz  * uz * ux +
			     dlx  * dlymS * dlz  * uz * uy +
			     dlx  * dlym  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = (dlxS * dlym  * dlzm  * uz * ux +
			     dlx  * dlymS * dlzm  * uz * uy +
			     dlx  * dlym  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = (dlxmS * dlyp  * dlzp  * uz * ux +
			     dlxm  * dlypS * dlzp  * uz * uy +
			     dlxm  * dlyp  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = (dlxmS * dlyp  * dlz  * uz * ux +
			     dlxm  * dlypS * dlz  * uz * uy +
			     dlxm  * dlyp  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = (dlxmS * dlyp  * dlzm  * uz * ux +
			     dlxm  * dlypS * dlzm  * uz * uy +
			     dlxm  * dlyp  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = (dlxmS * dly  * dlzp  * uz * ux +
			     dlxm  * dlyS * dlzp  * uz * uy +
			     dlxm  * dly  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = (dlxmS * dly  * dlz  * uz * ux +
			     dlxm  * dlyS * dlz  * uz * uy +
			     dlxm  * dly  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = (dlxmS * dly  * dlzm  * uz * ux +
			     dlxm  * dlyS * dlzm  * uz * uy +
			     dlxm  * dly  * dlzmS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = (dlxmS * dlym  * dlzp  * uz * ux +
			     dlxm  * dlymS * dlzp  * uz * uy +
			     dlxm  * dlym  * dlzpS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = (dlxmS * dlym  * dlz  * uz * ux +
			     dlxm  * dlymS * dlz  * uz * uy +
			     dlxm  * dlym  * dlzS * uz * uz) * massParticleGPU;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = (dlxmS * dlym  * dlzm  * uz * ux +
			     dlxm  * dlymS * dlzm  * uz * uy +
			     dlxm  * dlym  * dlzmS * uz * uz) * massParticleGPU;

  
}























//Add the particle advection and save in vxPredictionGPU
__global__ void kernelAddParticleAdvection(double* vxPredictionGPU,
					   double* vyPredictionGPU,
					   double* vzPredictionGPU,
					   double* fxboundaryGPU,
					   double* fyboundaryGPU,
					   double* fzboundaryGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  int particle, vecino;
  double fx = 0.f;
  double fy = 0.f;
  double fz = 0.f;


  //Particle contribution
  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+i);
    fz -= fzboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = tex1Dfetch(texvecino0GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
      fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
      fz -= fzboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }  
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino5
  vecino = tex1Dfetch(texvecino5GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = tex1Dfetch(texvecinopxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = tex1Dfetch(texvecinopxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = tex1Dfetch(texvecinomxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = tex1Dfetch(texvecinomxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = tex1Dfetch(texvecinopypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = tex1Dfetch(texvecinopymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = tex1Dfetch(texvecinomypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = tex1Dfetch(texvecinomymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = tex1Dfetch(texvecinopxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = tex1Dfetch(texvecinopxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = tex1Dfetch(texvecinopxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = tex1Dfetch(texvecinopxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = tex1Dfetch(texvecinomxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = tex1Dfetch(texvecinomxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = tex1Dfetch(texvecinomxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = tex1Dfetch(texvecinomxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle];
  }


  vxPredictionGPU[i] = dtGPU * fx / (volumeGPU * densfluidGPU);
  vyPredictionGPU[i] = dtGPU * fy / (volumeGPU * densfluidGPU);
  vzPredictionGPU[i] = dtGPU * fz / (volumeGPU * densfluidGPU);


}













//Add the particle advection and save in vxPredictionGPU
//together with the fluid advection
__global__ void kernelAddParticleAdvection_2(double* vxPredictionGPU,
					     double* vyPredictionGPU,
					     double* vzPredictionGPU,
					     cufftDoubleComplex* vxZ,
					     cufftDoubleComplex* vyZ,
					     cufftDoubleComplex* vzZ,
					     double* fxboundaryGPU,
					     double* fyboundaryGPU,
					     double* fzboundaryGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  int particle, vecino;
  double fx = 0.f;
  double fy = 0.f;
  double fz = 0.f;


  //Particle contribution
  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+i);
    fz -= fzboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = tex1Dfetch(texvecino0GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
      fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
      fz -= fzboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }  
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino5
  vecino = tex1Dfetch(texvecino5GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = tex1Dfetch(texvecinopxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = tex1Dfetch(texvecinopxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = tex1Dfetch(texvecinomxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = tex1Dfetch(texvecinomxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = tex1Dfetch(texvecinopypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = tex1Dfetch(texvecinopymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = tex1Dfetch(texvecinomypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = tex1Dfetch(texvecinomymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = tex1Dfetch(texvecinopxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = tex1Dfetch(texvecinopxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = tex1Dfetch(texvecinopxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = tex1Dfetch(texvecinopxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = tex1Dfetch(texvecinomxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = tex1Dfetch(texvecinomxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = tex1Dfetch(texvecinomxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = tex1Dfetch(texvecinomxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle];
  }


  //vxPredictionGPU[i] += dtGPU * fx / (volumeGPU * densfluidGPU);
  //vyPredictionGPU[i] += dtGPU * fy / (volumeGPU * densfluidGPU);
  //vzPredictionGPU[i] += dtGPU * fz / (volumeGPU * densfluidGPU);
  vxPredictionGPU[i] = vxZ[i].x + dtGPU * fx / (volumeGPU * densfluidGPU);
  vyPredictionGPU[i] = vyZ[i].x + dtGPU * fy / (volumeGPU * densfluidGPU);
  vzPredictionGPU[i] = vzZ[i].x + dtGPU * fz / (volumeGPU * densfluidGPU);

}













//Fill "countparticlesincellX" lists
//and spread particle force F 
__global__ void kernelSpreadParticlesForce(const double* rxcellGPU, 
					   const double* rycellGPU, 
					   const double* rzcellGPU,
					   double* fxboundaryGPU,
					   double* fyboundaryGPU,
					   double* fzboundaryGPU,
					   particlesincell* pc,
					   int* errorKernel,
					   const bondedForcesVariables* bFV){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double fx = 0.;//pressurea0GPU;// * 0.267261241912424385 ;//0;
  double fy = 0.;//pressurea0GPU * 0.534522483824848769 ;
  double fz = 0.;//pressurea0GPU * 0.801783725737273154 ;
  double f;
 

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  //INCLUDE EXTERNAL FORCES HERE

  //Example: harmonic potential 
  // V(r) = (1/2) * k * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
  //
  //with spring constant k=0.01
  //and x0=y0=z0=0
  //
  //fx = -0.01*rx;
  //fy = -0.01*ry;
  //fz = -0.01*rz;



  
  if(particlesWallGPU){
    //INCLUDE WALL REPULSION HERE
    //We use a repulsive Lennard-Jones
    double sigmaWall = 2*dyGPU;
    double cutoffWall = 1.12246204830937302 * sigmaWall; // 2^{1/6} * sigmaWall

    //IMPORTANT, for particleWall lyGPU stores ly+2*dy
    if(ry<(-0.5*lyGPU+cutoffWall+dyGPU)){//Left wall
      
      double distance = (0.5*lyGPU-dyGPU) + ry; //distance >= 0
      fy += 48 * temperatureGPU * (pow((sigmaWall/distance),13) - 0.5*pow((sigmaWall/distance),7));
      
    }
    else if(ry>(0.5*lyGPU-cutoffWall-dyGPU)){//Right wall
      
      double distance = (0.5*lyGPU-dyGPU) - ry; //distance >= 0
      fy -= 48 * temperatureGPU * (pow((sigmaWall/distance),13) - 0.5*pow((sigmaWall/distance),7));
      
    }
  }





  //NEW bonded forces
  if(bondedForcesGPU){
    //call function for bonded forces particle-particle
    forceBondedParticleParticleGPU(i,
				   fx,
				   fy,
				   fz,
				   rx,
				   ry,
				   rz,
				   bFV);
  }
    
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
  
  int np;
  if(computeNonBondedForcesGPU){
    //Particles in Cell i
    np = tex1Dfetch(texCountParticlesInCellNonBonded,icel);
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
    
  }
  
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

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
  pc->partincellX[ncellstGPU*np+icelx] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellY[icely],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[2]=np;
    return;
  }
  pc->partincellY[ncellstGPU*np+icely] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellZ[icelz],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[3]=np;
    return;
  }
  pc->partincellZ[ncellstGPU*np+icelz] = nboundaryGPU+i;

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

  //double dlxS, dlxpS, dlxmS;
  //double dlyS, dlypS, dlymS;
  //double dlzS, dlzpS, dlzmS;

  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx  = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  //dlxS = functionDeltaDerived(1.5*r);
  //dlxpS = functionDeltaDerived(1.5*rp);
  //dlxmS = functionDeltaDerived(1.5*rm);
                              
  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  //dlyS = functionDeltaDerived(1.5*r);
  //dlypS = functionDeltaDerived(1.5*rp);
  //dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  //dlzS = functionDeltaDerived(1.5*r);
  //dlzpS = functionDeltaDerived(1.5*rp);
  //dlzmS = functionDeltaDerived(1.5*rm);

    
  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fx;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fx;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fx;
  
  
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
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  //dlxS = functionDeltaDerived(1.5*r);
  //dlxpS = functionDeltaDerived(1.5*rp);
  //dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  //dlyS = functionDeltaDerived(1.5*r);
  //dlypS = functionDeltaDerived(1.5*rp);
  //dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  //dlzS = functionDeltaDerived(1.5*r);
  //dlzpS = functionDeltaDerived(1.5*rp);
  //dlzmS = functionDeltaDerived(1.5*rm);

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fy;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fy;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fy;


  
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
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  //dlxS = functionDeltaDerived(1.5*r);
  //dlxpS = functionDeltaDerived(1.5*rp);
  //dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);
  //dlyS = functionDeltaDerived(1.5*r);
  //dlypS = functionDeltaDerived(1.5*rp);
  //dlymS = functionDeltaDerived(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  //dlzS = functionDeltaDerived(1.5*r);
  //dlzpS = functionDeltaDerived(1.5*rp);
  //dlzmS = functionDeltaDerived(1.5*rm);


  offset = nboundaryGPU;
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fz;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fz;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fz;

  
}




