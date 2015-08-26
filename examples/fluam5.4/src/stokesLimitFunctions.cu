// Filename: stokesLimitFunctions.cu
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




__global__ void findNeighborParticlesStokesLimit_1(particlesincell* pc, 
						   int* errorKernel){
						   
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  



  int icel;
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    rx = rx - (int(rx*invlxGPU + 0.5*((rx>0)-(rx<0)))) * lxGPU;
    int jx   = int(rx * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    ry = ry - (int(ry*invlyGPU + 0.5*((ry>0)-(ry<0)))) * lyGPU;
    int jy   = int(ry * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    rz = rz - (int(rz*invlzGPU + 0.5*((rz>0)-(rz<0)))) * lzGPU;
    int jz   = int(rz * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }
  int np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;



}



















//Fill "countparticlesincellX" lists
//and spread particle force F 
__global__ void kernelSpreadParticlesForceStokesLimit(const double* rxcellGPU, 
						      const double* rycellGPU, 
						      const double* rzcellGPU,
						      double* fxboundaryGPU,
						      double* fyboundaryGPU,
						      double* fzboundaryGPU,
						      double* vxboundaryGPU,//Fx, this is the force on the particle.
						      double* vyboundaryGPU,
						      double* vzboundaryGPU,
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

  vxboundaryGPU[nboundaryGPU+i] = fx; //Fx, this is the force on the particle.
  vyboundaryGPU[nboundaryGPU+i] = fy;
  vzboundaryGPU[nboundaryGPU+i] = fz;
  
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







































__global__ void addSpreadedForcesStokesLimit(double* vxGPU, //fx=S*Fx
					     double* vyGPU, 
					     double* vzGPU,
					     const double* fxboundaryGPU, 
					     const double* fyboundaryGPU, 
					     const double* fzboundaryGPU,
					     const particlesincell* pc){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellstGPU) return;

  int vecino, particle;
  double fx = 0.f;
  double fy = 0.f;
  double fz = 0.f;

  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+i);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+i);
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+i);
    fz -= fzboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = tex1Dfetch(texvecino0GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
      fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
      fz -= fzboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }  
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino5
  vecino = tex1Dfetch(texvecino5GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = tex1Dfetch(texvecinopxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = tex1Dfetch(texvecinopxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = tex1Dfetch(texvecinomxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = tex1Dfetch(texvecinomxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = tex1Dfetch(texvecinopypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = tex1Dfetch(texvecinopymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = tex1Dfetch(texvecinomypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = tex1Dfetch(texvecinomymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = tex1Dfetch(texvecinopxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = tex1Dfetch(texvecinopxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = tex1Dfetch(texvecinopxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = tex1Dfetch(texvecinopxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = tex1Dfetch(texvecinomxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = tex1Dfetch(texvecinomxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = tex1Dfetch(texvecinomxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = tex1Dfetch(texvecinomxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellstGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellstGPU*j+vecino);
    fz -= fzboundaryGPU[particle];
  }


  
  vxGPU[i] += fx;
  vyGPU[i] += fy;
  vzGPU[i] += fz;

}























__global__ void kernelSpreadParticlesDrift(const double* rxcellGPU, 
					   const double* rycellGPU, 
					   const double* rzcellGPU,
					   double* fxboundaryGPU,
					   double* fyboundaryGPU,
					   double* fzboundaryGPU,
					   const double* d_rand,
					   particlesincell* pc,
					   int sign){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double deltaDisplacement = 0.001;
  double f = (temperatureGPU / deltaDisplacement) * sign;
  double fx = f * d_rand[6*ncellsGPU +           i];
  double fy = f * d_rand[6*ncellsGPU +   npGPU + i];
  double fz = f * d_rand[6*ncellsGPU + 2*npGPU + i];

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

  //Displacement to positive values
  double rxR = rx + sign * 0.5 * deltaDisplacement * d_rand[6*ncellsGPU +           i];
  double ryR = ry + sign * 0.5 * deltaDisplacement * d_rand[6*ncellsGPU +   npGPU + i];
  double rzR = rz + sign * 0.5 * deltaDisplacement * d_rand[6*ncellsGPU + 2*npGPU + i];

  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    r = rxR;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ryR;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    r = rzR;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }
  

  
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rxR;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rxR - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ryR;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ryR - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    r = rzR;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rzR - 0.5*dzGPU;
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

  //THERMAL DRIFT IN THE X DIRECTION
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

























__global__ void kernelConstructWstokesLimit(const double *vxGPU, //Stored fx=S*Fx+S*drift_p-S*drift_m
					    const double *vyGPU, 
					    const double *vzGPU, 
					    cufftDoubleComplex *WxZ, 
					    cufftDoubleComplex *WyZ, 
					    cufftDoubleComplex *WzZ, 
					    const double* d_rand){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy, wz;
  double vx, vy, vz;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 

  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);

  //Stored fx=S*Fx+S*drift_p-S*drift_m
  wx = vx;
  wy = vy;
  wz = vz;

  

  //NOISE part
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;

  dnoise_sXX = d_rand[vecino3];
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU];
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU];
  wz += invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4GPU * dnoise_sXY;
  wy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  wx += invdzGPU * fact4GPU * dnoise_sXZ;
  wz += invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 4*ncellsGPU];
  wy += invdzGPU * fact4GPU * dnoise_sYZ;
  wz += invdyGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXX = d_rand[i];
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU];
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU];
  wz -= invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  wx -= invdzGPU * fact4GPU * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 4*ncellsGPU];
  wy -= invdzGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 2*ncellsGPU];
  wz -= invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 4*ncellsGPU];
  wz -= invdyGPU * fact4GPU * dnoise_sYZ;


  WxZ[i].x = wx;
  WyZ[i].x = wy;
  WzZ[i].x = wz;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}




















__global__ void kernelUpdateVstokesLimit(cufftDoubleComplex *vxZ, 
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
  double denominator = - shearviscosityGPU * L;

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



























__global__ void updateParticlesStokesLimit_1(particlesincell* pc, 
					     int* errorKernel,
					     const double* rxcellGPU,
					     const double* rycellGPU,
					     const double* rzcellGPU,
					     const double* rxboundaryGPU,  //q^{n}
					     const double* ryboundaryGPU, 
					     const double* rzboundaryGPU,
					     double* rxboundaryPredictionGPU, //q^{n+1/2}
					     double* ryboundaryPredictionGPU, 
					     double* rzboundaryPredictionGPU,
					     const double* vxboundaryGPU, //Fx^n
					     const double* vyboundaryGPU, 
					     const double* vzboundaryGPU, 
					     const double* vxGPU, //v^n
					     const double* vyGPU, 
					     const double* vzGPU,
					     const double* d_rand){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;
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
  
  //VELOCITY IN THE X DIRECTION
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
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  
  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  
  double v = dlxm * dlym * dlzm * vxGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vxGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vxGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vxGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vxGPU[vecino2] +
    dlxm * dly  * dlzp * vxGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vxGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vxGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vxGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vxGPU[vecinomymz] +
    dlx  * dlym * dlz  * vxGPU[vecino1] +
    dlx  * dlym * dlzp * vxGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vxGPU[vecino0] + 
    dlx  * dly  * dlz  * vxGPU[icelx] + 
    dlx  * dly  * dlzp * vxGPU[vecino5] + 
    dlx  * dlyp * dlzm * vxGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vxGPU[vecino4] + 
    dlx  * dlyp * dlzp * vxGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vxGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vxGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vxGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vxGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vxGPU[vecino3] +
    dlxp * dly  * dlzp * vxGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vxGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vxGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vxGPU[vecinopxpypz];

  

  double rxNew = rx + 0.5 * dtGPU * v;
  if(setExtraMobilityGPU){ 
    rxNew += 0.5*dtGPU*extraMobilityGPU * vxboundaryGPU[nboundaryGPU+i]
      + fact2GPU * d_rand[6*ncellsGPU + 3*npGPU + i];
  }
  rxboundaryPredictionGPU[nboundaryGPU+i] = rxNew;


  //VELOCITY IN THE Y DIRECTION
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
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  v = dlxm * dlym * dlzm * vyGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vyGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vyGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vyGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vyGPU[vecino2] +
    dlxm * dly  * dlzp * vyGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vyGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vyGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vyGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vyGPU[vecinomymz] +
    dlx  * dlym * dlz  * vyGPU[vecino1] +
    dlx  * dlym * dlzp * vyGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vyGPU[vecino0] + 
    dlx  * dly  * dlz  * vyGPU[icely] + 
    dlx  * dly  * dlzp * vyGPU[vecino5] + 
    dlx  * dlyp * dlzm * vyGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vyGPU[vecino4] + 
    dlx  * dlyp * dlzp * vyGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vyGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vyGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vyGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vyGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vyGPU[vecino3] +
    dlxp * dly  * dlzp * vyGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vyGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vyGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vyGPU[vecinopxpypz];
  

  double ryNew = ry + 0.5 * dtGPU * v;
  if(setExtraMobilityGPU){
    ryNew += 0.5*dtGPU*extraMobilityGPU * vyboundaryGPU[nboundaryGPU+i]
      + fact2GPU * d_rand[6*ncellsGPU + 4*npGPU + i];
  }
  ryboundaryPredictionGPU[nboundaryGPU+i] = ryNew;
 
  //VELOCITY IN THE Z DIRECTION
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
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

 
  v = dlxm * dlym * dlzm * vzGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vzGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vzGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vzGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vzGPU[vecino2] +
    dlxm * dly  * dlzp * vzGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vzGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vzGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vzGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vzGPU[vecinomymz] +
    dlx  * dlym * dlz  * vzGPU[vecino1] +
    dlx  * dlym * dlzp * vzGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vzGPU[vecino0] + 
    dlx  * dly  * dlz  * vzGPU[icelz] + 
    dlx  * dly  * dlzp * vzGPU[vecino5] + 
    dlx  * dlyp * dlzm * vzGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vzGPU[vecino4] + 
    dlx  * dlyp * dlzp * vzGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vzGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vzGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vzGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vzGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vzGPU[vecino3] +
    dlxp * dly  * dlzp * vzGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vzGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vzGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vzGPU[vecinopxpypz];



  double rzNew = rz + 0.5 * dtGPU * v;
  if(setExtraMobilityGPU){
    rzNew += 0.5*dtGPU*extraMobilityGPU * vzboundaryGPU[nboundaryGPU+i]
      + fact2GPU * d_rand[6*ncellsGPU + 5*npGPU + i];
  }
  rzboundaryPredictionGPU[nboundaryGPU+i] = rzNew;



  int icel;
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    rxNew = rxNew - (int(rxNew*invlxGPU + 0.5*((rxNew>0)-(rxNew<0)))) * lxGPU;
    int jx   = int(rxNew * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    ryNew = ryNew - (int(ryNew*invlyGPU + 0.5*((ryNew>0)-(ryNew<0)))) * lyGPU;
    int jy   = int(ryNew * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    rzNew = rzNew - (int(rzNew*invlzGPU + 0.5*((rzNew>0)-(rzNew<0)))) * lzGPU;
    int jz   = int(rzNew * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }
  int np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;



}























__global__ void particlesForceStokesLimit(double* vxboundaryGPU,//Fx, this is the force on the particle.
					  double* vyboundaryGPU,
					  double* vzboundaryGPU,
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
  
  vxboundaryGPU[nboundaryGPU+i] = fx; //Fx, this is the force on the particle.
  vyboundaryGPU[nboundaryGPU+i] = fy;
  vzboundaryGPU[nboundaryGPU+i] = fz;
  
  
}

























__global__ void updateParticlesStokesLimit_2(const double* rxcellGPU,
					     const double* rycellGPU,
					     const double* rzcellGPU,
					     double* rxboundaryGPU,  //q^{n}
					     double* ryboundaryGPU, 
					     double* rzboundaryGPU,
					     const double* vxboundaryGPU, //Fx^n
					     const double* vyboundaryGPU, 
					     const double* vzboundaryGPU, 
					     const double* vxGPU, //v^n
					     const double* vyGPU, 
					     const double* vzGPU,
					     const double* d_rand,
					     particlesincell* pc, 
					     int* errorKernel){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;
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
  
  //VELOCITY IN THE X DIRECTION
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
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  
  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  
  double v = dlxm * dlym * dlzm * vxGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vxGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vxGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vxGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vxGPU[vecino2] +
    dlxm * dly  * dlzp * vxGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vxGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vxGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vxGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vxGPU[vecinomymz] +
    dlx  * dlym * dlz  * vxGPU[vecino1] +
    dlx  * dlym * dlzp * vxGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vxGPU[vecino0] + 
    dlx  * dly  * dlz  * vxGPU[icelx] + 
    dlx  * dly  * dlzp * vxGPU[vecino5] + 
    dlx  * dlyp * dlzm * vxGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vxGPU[vecino4] + 
    dlx  * dlyp * dlzp * vxGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vxGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vxGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vxGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vxGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vxGPU[vecino3] +
    dlxp * dly  * dlzp * vxGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vxGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vxGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vxGPU[vecinopxpypz];


  double rxNew = rxboundaryGPU[nboundaryGPU+i] + dtGPU * v;
  if(setExtraMobilityGPU){
    rxNew += dtGPU*extraMobilityGPU * vxboundaryGPU[nboundaryGPU+i]
      + fact2GPU * (d_rand[6*ncellsGPU + 3*npGPU + i] + d_rand[6*ncellsGPU + 6*npGPU + i]);
  }
  rxboundaryGPU[nboundaryGPU+i] = rxNew;


  //VELOCITY IN THE Y DIRECTION
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
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  v = dlxm * dlym * dlzm * vyGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vyGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vyGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vyGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vyGPU[vecino2] +
    dlxm * dly  * dlzp * vyGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vyGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vyGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vyGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vyGPU[vecinomymz] +
    dlx  * dlym * dlz  * vyGPU[vecino1] +
    dlx  * dlym * dlzp * vyGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vyGPU[vecino0] + 
    dlx  * dly  * dlz  * vyGPU[icely] + 
    dlx  * dly  * dlzp * vyGPU[vecino5] + 
    dlx  * dlyp * dlzm * vyGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vyGPU[vecino4] + 
    dlx  * dlyp * dlzp * vyGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vyGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vyGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vyGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vyGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vyGPU[vecino3] +
    dlxp * dly  * dlzp * vyGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vyGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vyGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vyGPU[vecinopxpypz];
  

  double ryNew = ryboundaryGPU[nboundaryGPU+i] + dtGPU * v;
  if(setExtraMobilityGPU){
    ryNew += dtGPU*extraMobilityGPU * vyboundaryGPU[nboundaryGPU+i]
      + fact2GPU * (d_rand[6*ncellsGPU + 4*npGPU + i] + d_rand[6*ncellsGPU + 7*npGPU + i]);
  }
  ryboundaryGPU[nboundaryGPU+i] = ryNew;
 
  //VELOCITY IN THE Z DIRECTION
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
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

 
  v = dlxm * dlym * dlzm * vzGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vzGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vzGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vzGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vzGPU[vecino2] +
    dlxm * dly  * dlzp * vzGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vzGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vzGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vzGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vzGPU[vecinomymz] +
    dlx  * dlym * dlz  * vzGPU[vecino1] +
    dlx  * dlym * dlzp * vzGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vzGPU[vecino0] + 
    dlx  * dly  * dlz  * vzGPU[icelz] + 
    dlx  * dly  * dlzp * vzGPU[vecino5] + 
    dlx  * dlyp * dlzm * vzGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vzGPU[vecino4] + 
    dlx  * dlyp * dlzp * vzGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vzGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vzGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vzGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vzGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vzGPU[vecino3] +
    dlxp * dly  * dlzp * vzGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vzGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vzGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vzGPU[vecinopxpypz];


  double rzNew = rzboundaryGPU[nboundaryGPU+i] + dtGPU * v;
  if(setExtraMobilityGPU){
    rzNew += dtGPU*extraMobilityGPU * vzboundaryGPU[nboundaryGPU+i]
      + fact2GPU * (d_rand[6*ncellsGPU + 5*npGPU + i] + d_rand[6*ncellsGPU + 8*npGPU + i]);           
  }
  rzboundaryGPU[nboundaryGPU+i] = rzNew;

}


