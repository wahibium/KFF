// Filename: nonBondedForceCompressibleParticlesExtraPressure.cu
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





//Calculate and spread "pressure" due to the particle S*Omega
__global__ void kernelSpreadPressureParticles(double* rxcellGPU,
					      double* rycellGPU,
					      double* rzcellGPU,
					      double* densityGPU,
					      double* fboundaryOmega,
					      particlesincell* pc){
  
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
  
  //int icel;
  double r, rp, rm;

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icel;

  {
    int mxmy = mxGPU * mytGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icel   = jx;
    icel  += jy * mxGPU;
    icel  += jz * mxmy;
  }

  int np = atomicAdd(&pc->countparticlesincell[icel],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }  
  pc->partincell[ncellstGPU*np+icel] = nboundaryGPU+i;

  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;

  //DENSITY
  vecino0 = tex1Dfetch(texvecino0GPU, icel);
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecino5 = tex1Dfetch(texvecino5GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icel);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icel);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icel);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icel);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icel);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icel);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icel);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icel);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icel);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icel);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icel);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icel);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icel);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icel);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icel);

  r =  (rx - rxcellGPU[icel]);
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

  r =  (ry - rycellGPU[icel]);
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

  r =  (rz - rzcellGPU[icel]);
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

  double density = dlxm * dlym * dlzm * (densityGPU[vecinomxmymz]) +
    dlxm * dlym * dlz  * (densityGPU[vecinomxmy]) +
    dlxm * dlym * dlzp * (densityGPU[vecinomxmypz]) +
    dlxm * dly  * dlzm * (densityGPU[vecinomxmz]) + 
    dlxm * dly  * dlz  * (densityGPU[vecino2]) +
    dlxm * dly  * dlzp * (densityGPU[vecinomxpz]) +
    dlxm * dlyp * dlzm * (densityGPU[vecinomxpymz]) +
    dlxm * dlyp * dlz  * (densityGPU[vecinomxpy]) +
    dlxm * dlyp * dlzp * (densityGPU[vecinomxpypz]) +
    dlx  * dlym * dlzm * (densityGPU[vecinomymz]) +
    dlx  * dlym * dlz  * (densityGPU[vecino1]) +
    dlx  * dlym * dlzp * (densityGPU[vecinomypz]) + 
    dlx  * dly  * dlzm * (densityGPU[vecino0]) + 
    dlx  * dly  * dlz  * (densityGPU[icel]) + 
    dlx  * dly  * dlzp * (densityGPU[vecino5]) + 
    dlx  * dlyp * dlzm * (densityGPU[vecinopymz]) + 
    dlx  * dlyp * dlz  * (densityGPU[vecino4]) + 
    dlx  * dlyp * dlzp * (densityGPU[vecinopypz]) + 
    dlxp * dlym * dlzm * (densityGPU[vecinopxmymz]) +
    dlxp * dlym * dlz  * (densityGPU[vecinopxmy]) + 
    dlxp * dlym * dlzp * (densityGPU[vecinopxmypz]) +
    dlxp * dly  * dlzm * (densityGPU[vecinopxmz]) + 
    dlxp * dly  * dlz  * (densityGPU[vecino3]) + 
    dlxp * dly  * dlzp * (densityGPU[vecinopxpz]) +
    dlxp * dlyp * dlzm * (densityGPU[vecinopxpymz]) +
    dlxp * dlyp * dlz  * (densityGPU[vecinopxpy]) + 
    dlxp * dlyp * dlzp * (densityGPU[vecinopxpypz]);
  


  double f = omega0GPU * (density - densfluidGPU);


  int offset = nboundaryGPU;
  fboundaryOmega[offset+i]        = dlxp * dlyp * dlzp * f ;
  offset += nboundaryGPU+npGPU;//1
  fboundaryOmega[offset+i] = dlxp * dlyp * dlz  * f ;
  offset += nboundaryGPU+npGPU;//2
  fboundaryOmega[offset+i] = dlxp * dlyp * dlzm * f ;
  offset += nboundaryGPU+npGPU;//3
  fboundaryOmega[offset+i] = dlxp * dly  * dlzp * f ;
  offset += nboundaryGPU+npGPU;//4
  fboundaryOmega[offset+i] = dlxp * dly  * dlz  * f ;
  offset += nboundaryGPU+npGPU;//5
  fboundaryOmega[offset+i] = dlxp * dly  * dlzm * f ;
  offset += nboundaryGPU+npGPU;//6
  fboundaryOmega[offset+i] = dlxp * dlym * dlzp * f ;
  offset += nboundaryGPU+npGPU;//7
  fboundaryOmega[offset+i] = dlxp * dlym * dlz  * f ;
  offset += nboundaryGPU+npGPU;//8
  fboundaryOmega[offset+i] = dlxp * dlym * dlzm * f ;
  offset += nboundaryGPU+npGPU;//9
  fboundaryOmega[offset+i] = dlx  * dlyp * dlzp * f ;
  offset += nboundaryGPU+npGPU;//10
  fboundaryOmega[offset+i] = dlx  * dlyp * dlz  * f ;
  offset += nboundaryGPU+npGPU;//11
  fboundaryOmega[offset+i] = dlx  * dlyp * dlzm * f ;
  offset += nboundaryGPU+npGPU;//12
  fboundaryOmega[offset+i] = dlx  * dly  * dlzp * f ;
  offset += nboundaryGPU+npGPU;//13
  fboundaryOmega[offset+i] = dlx  * dly  * dlz  * f ;
  offset += nboundaryGPU+npGPU;//14
  fboundaryOmega[offset+i] = dlx  * dly  * dlzm * f ;
  offset += nboundaryGPU+npGPU;//15
  fboundaryOmega[offset+i] = dlx  * dlym * dlzp * f ;
  offset += nboundaryGPU+npGPU;//16
  fboundaryOmega[offset+i] = dlx  * dlym * dlz  * f ;
  offset += nboundaryGPU+npGPU;//17
  fboundaryOmega[offset+i] = dlx  * dlym * dlzm * f ;
  offset += nboundaryGPU+npGPU;//18
  fboundaryOmega[offset+i] = dlxm * dlyp * dlzp * f ;
  offset += nboundaryGPU+npGPU;//19
  fboundaryOmega[offset+i] = dlxm * dlyp * dlz  * f ;
  offset += nboundaryGPU+npGPU;//20
  fboundaryOmega[offset+i] = dlxm * dlyp * dlzm * f ;
  offset += nboundaryGPU+npGPU;//21
  fboundaryOmega[offset+i] = dlxm * dly  * dlzp * f ;
  offset += nboundaryGPU+npGPU;//22
  fboundaryOmega[offset+i] = dlxm * dly  * dlz  * f ;
  offset += nboundaryGPU+npGPU;//23
  fboundaryOmega[offset+i] = dlxm * dly  * dlzm * f ;
  offset += nboundaryGPU+npGPU;//24
  fboundaryOmega[offset+i] = dlxm * dlym * dlzp * f ;
  offset += nboundaryGPU+npGPU;//25
  fboundaryOmega[offset+i] = dlxm * dlym * dlz  * f ;
  offset += nboundaryGPU+npGPU;//26
  fboundaryOmega[offset+i] = dlxm * dlym * dlzm * f ;






}
      












__global__ void kernelAddPressureParticles(double *omegaGPU,
					   double *fboundaryOmega,
					   particlesincell* pc){

  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  int vecino, particle;
  double fx = 0.;


  //Particles in Cell i
  //int np = tex1Dfetch(texCountParticlesInCellX,i);
  int np = pc->countparticlesincell[i];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+i);
    particle = pc->partincell[ncellstGPU*j+i];
    fx += fboundaryOmega[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = tex1Dfetch(texvecino0GPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+12*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+4*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+22*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+16*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino5
  vecino = tex1Dfetch(texvecino5GPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = tex1Dfetch(texvecinopxpzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = tex1Dfetch(texvecinopxmzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = tex1Dfetch(texvecinomxpzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = tex1Dfetch(texvecinomxmzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = tex1Dfetch(texvecinopypzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = tex1Dfetch(texvecinopymzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = tex1Dfetch(texvecinomypzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = tex1Dfetch(texvecinomymzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = tex1Dfetch(texvecinopxpypzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = tex1Dfetch(texvecinopxpymzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = tex1Dfetch(texvecinopxmypzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = tex1Dfetch(texvecinopxmymzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = tex1Dfetch(texvecinomxpypzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = tex1Dfetch(texvecinomxpymzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = tex1Dfetch(texvecinomxmypzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = tex1Dfetch(texvecinomxmymzGPU, i);
  //np = tex1Dfetch(texCountParticlesInCellX,vecino);
  np = pc->countparticlesincell[vecino];
  for(int j=0;j<np;j++){
    //particle = tex1Dfetch(texPartInCellX,ncellstGPU*j+vecino);
    particle = pc->partincell[ncellstGPU*j+vecino];
    fx += fboundaryOmega[particle];
  }

  
  //omegaGPU[i] = fx / (volumeGPU * volumeParticleGPU);
  omegaGPU[i] = fx / (volumeGPU);


}















__global__ void countToZeroListX(particlesincell* pc){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(i<ncellsGPU){
    pc->countparticlesincell[i] = 0;
  }
  
  
}




