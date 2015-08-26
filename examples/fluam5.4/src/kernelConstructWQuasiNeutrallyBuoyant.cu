// Filename: kernelConstructWQuasiNeutrallyBuoyant.cu
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



//In this kernel we construct the vector W
//
// W = v^n + 0.5*dt*nu*L*v^n + Advection^{n+1/2} + (dt/rho)*f^n_{noise} + dt*SF/rho 
//
__global__ void kernelConstructWQuasiNeutrallyBuoyant(const double *vxPredictionGPU, 
						      const double *vyPredictionGPU, 
						      const double *vzPredictionGPU, 
						      cufftDoubleComplex *WxZ, 
						      cufftDoubleComplex *WyZ, 
						      cufftDoubleComplex *WzZ, 
						      const double *d_rand,
						      const double *fxboundaryGPU,
						      const double *fyboundaryGPU,
						      const double *fzboundaryGPU,
						      double* advXGPU,
						      double* advYGPU,
						      double* advZGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy, wz;
  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  int vecinopxmy, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopymz, vecinomypz;
  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;

  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);



  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx += invdzGPU * invdzGPU * (vx5 - 2*vx + vx0);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy += invdzGPU * invdzGPU * (vy5 - 2*vy + vy0);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;
  wz  = invdxGPU * invdxGPU * (vz3 - 2*vz + vz2);
  wz += invdyGPU * invdyGPU * (vz4 - 2*vz + vz1);
  wz += invdzGPU * invdzGPU * (vz5 - 2*vz + vz0);
  wz  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wz;

  //Previous Velocity
  wx += vx ;//- pressurea1GPU * dtGPU;// * 0.267261241912424385 * dtGPU;
  wy += vy ;//- pressurea1GPU * 0.534522483824848769 * dtGPU;
  wz += vz ;//- pressurea1GPU * 0.801783725737273154 * dtGPU;
  
  //Advection part
  double advX, advY, advZ;
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX += invdzGPU * ((vx5+vx)*(vz3+vz) - (vx+vx0)*(vzpxmz+vz0));
  advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY += invdzGPU * ((vy5+vy)*(vz4+vz) - (vy+vy0)*(vzpymz+vz0));
  advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(vx5+vx) - (vz+vz2)*(vxmxpz+vx2));
  advZ += invdyGPU * ((vz4+vz)*(vy5+vy) - (vz+vz1)*(vymypz+vy1));
  advZ += invdzGPU * ((vz5+vz)*(vz5+vz) - (vz+vz0)*(vz+vz0));
  advZ  = 0.25 * dtGPU * advZ;
  
  //advX=0; advY=0; advZ=0;
  //Add particle's advection
  //24-10-2012, we should not add the particle kinetic stress twice!
  //advX += vxPredictionGPU[i];
  //advY += vyPredictionGPU[i];
  //advZ += vzPredictionGPU[i];
  

  //NOISE part
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 3*ncellsGPU] + d_rand[vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/3.;
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 3*ncellsGPU] + d_rand[vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_tr = d_rand[vecino5] + d_rand[vecino5 + 3*ncellsGPU] + d_rand[vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
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

  dnoise_tr = d_rand[i] + d_rand[i + 3*ncellsGPU] + d_rand[i + 5*ncellsGPU];
  dnoise_sXX = d_rand[i] - dnoise_tr/3.;
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU] - dnoise_tr/3.;
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU] - dnoise_tr/3.;
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
  

  int vecino, particle;
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




  WxZ[i].x = wx + dtGPU * ( fx / (volumeGPU*densfluidGPU) ) - advX  ;
  WyZ[i].x = wy + dtGPU * ( fy / (volumeGPU*densfluidGPU) ) - advY  ;
  WzZ[i].x = wz + dtGPU * ( fz / (volumeGPU*densfluidGPU) ) - advZ  ;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

  //Save advection for the next time step
  advXGPU[i] = advX ;
  advYGPU[i] = advY ;
  advZGPU[i] = advZ ;

}
















//In this kernel we construct the vector W
//
// W = v^n + 0.5*dt*nu*L*v^n + Advection^{n+1/2} + (dt/rho)*f^n_{noise} + dt*SF/rho 
//


__global__ void kernelConstructWQuasiNeutrallyBuoyantTEST5_2(double *vxPredictionGPU, 
							     double *vyPredictionGPU, 
							     double *vzPredictionGPU, 
							     cufftDoubleComplex *WxZ, 
							     cufftDoubleComplex *WyZ, 
							     cufftDoubleComplex *WzZ, 
							     double *d_rand,
							     double *fxboundaryGPU,
							     double *fyboundaryGPU,
							     double *fzboundaryGPU,
							     double* advXGPU,
							     double* advYGPU,
							     double* advZGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy, wz;
  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
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
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);
  


  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx += invdzGPU * invdzGPU * (vx5 - 2*vx + vx0);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy += invdzGPU * invdzGPU * (vy5 - 2*vy + vy0);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;
  wz  = invdxGPU * invdxGPU * (vz3 - 2*vz + vz2);
  wz += invdyGPU * invdyGPU * (vz4 - 2*vz + vz1);
  wz += invdzGPU * invdzGPU * (vz5 - 2*vz + vz0);
  wz  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wz;
  
  //Previous Velocity
  wx += vx ;//- pressurea1GPU * dtGPU;//* 0.267261241912424385 * dtGPU;
  wy += vy ;//- pressurea1GPU * 0.534522483824848769 * dtGPU;
  wz += vz ;//- pressurea1GPU * 0.801783725737273154 * dtGPU;
  
  //Advection part
  double advX, advY, advZ;
  //advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  //advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  //advX += invdzGPU * ((vx5+vx)*(vz3+vz) - (vx+vx0)*(vzpxmz+vz0));
  //advX  = 0.25 * dtGPU * advX;
  //advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  //advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  //advY += invdzGPU * ((vy5+vy)*(vz4+vz) - (vy+vy0)*(vzpymz+vz0));
  //advY  = 0.25 * dtGPU * advY;
  //advZ  = invdxGPU * ((vz3+vz)*(vx5+vx) - (vz+vz2)*(vxmxpz+vx2));
  //advZ += invdyGPU * ((vz4+vz)*(vy5+vy) - (vz+vz1)*(vymypz+vy1));
  //advZ += invdzGPU * ((vz5+vz)*(vz5+vz) - (vz+vz0)*(vz+vz0));
  //advZ  = 0.25 * dtGPU * advZ;

  //Advection at mid-point
  //24-10-2012, we should not add the particle kinetic stress twice!
  //advX = vxPredictionGPU[i];
  //advY = vyPredictionGPU[i];
  //advZ = vzPredictionGPU[i];
  advX = WxZ[i].x;
  advY = WyZ[i].x;
  advZ = WzZ[i].x;

  //TEST 24-2-2011
  //advX=0; advY=0; advZ=0;////////////
  //wx -= advX;
  //wy -= advY;
  //wz -= advZ;

  //NOISE part
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 3*ncellsGPU] + d_rand[vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/3.;
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 3*ncellsGPU] + d_rand[vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_tr = d_rand[vecino5] + d_rand[vecino5 + 3*ncellsGPU] + d_rand[vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
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

  dnoise_tr = d_rand[i] + d_rand[i + 3*ncellsGPU] + d_rand[i + 5*ncellsGPU];
  dnoise_sXX = d_rand[i] - dnoise_tr/3.;
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU] - dnoise_tr/3.;
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU] - dnoise_tr/3.;
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
 

  int vecino, particle;
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




  WxZ[i].x = wx + dtGPU * ( fx / (volumeGPU*densfluidGPU) ) - advX  ;
  WyZ[i].x = wy + dtGPU * ( fy / (volumeGPU*densfluidGPU) ) - advY  ;
  WzZ[i].x = wz + dtGPU * ( fz / (volumeGPU*densfluidGPU) ) - advZ  ;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;



}
















__global__ void kernelConstructWQuasiNeutrallyBuoyantTEST5_3(const double *vxPredictionGPU, 
							     const double *vyPredictionGPU, 
							     const double *vzPredictionGPU, 
							     cufftDoubleComplex *WxZ, 
							     cufftDoubleComplex *WyZ, 
							     cufftDoubleComplex *WzZ, 
							     const double *d_rand,
							     const double *fxboundaryGPU,
							     const double *fyboundaryGPU,
							     const double *fzboundaryGPU,
							     double* advXGPU,
							     double* advYGPU,
							     double* advZGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy, wz;
  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  int vecinopxmy, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopymz, vecinomypz;
  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;

  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);



  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx += invdzGPU * invdzGPU * (vx5 - 2*vx + vx0);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy += invdzGPU * invdzGPU * (vy5 - 2*vy + vy0);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;
  wz  = invdxGPU * invdxGPU * (vz3 - 2*vz + vz2);
  wz += invdyGPU * invdyGPU * (vz4 - 2*vz + vz1);
  wz += invdzGPU * invdzGPU * (vz5 - 2*vz + vz0);
  wz  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wz;

  //Previous Velocity
  wx += vx ;//- pressurea1GPU * dtGPU;//* 0.267261241912424385 * dtGPU;
  wy += vy ;//- pressurea1GPU * 0.534522483824848769 * dtGPU;
  wz += vz ;//- pressurea1GPU * 0.801783725737273154 * dtGPU;
  
  //Advection part
  double advX, advY, advZ;
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX += invdzGPU * ((vx5+vx)*(vz3+vz) - (vx+vx0)*(vzpxmz+vz0));
  advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY += invdzGPU * ((vy5+vy)*(vz4+vz) - (vy+vy0)*(vzpymz+vz0));
  advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(vx5+vx) - (vz+vz2)*(vxmxpz+vx2));
  advZ += invdyGPU * ((vz4+vz)*(vy5+vy) - (vz+vz1)*(vymypz+vy1));
  advZ += invdzGPU * ((vz5+vz)*(vz5+vz) - (vz+vz0)*(vz+vz0));
  advZ  = 0.25 * dtGPU * advZ;

  //Add particle's advection
  //24-10-2012, we should not add the particle kinetic stress twice!
  //advX += vxPredictionGPU[i];
  //advY += vyPredictionGPU[i];
  //advZ += vzPredictionGPU[i];

  //advX=0; advY=0; advZ=0;
  //NOISE part
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 3*ncellsGPU] + d_rand[vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/3.;
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 3*ncellsGPU] + d_rand[vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_tr = d_rand[vecino5] + d_rand[vecino5 + 3*ncellsGPU] + d_rand[vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
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

  dnoise_tr = d_rand[i] + d_rand[i + 3*ncellsGPU] + d_rand[i + 5*ncellsGPU];
  dnoise_sXX = d_rand[i] - dnoise_tr/3.;
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU] - dnoise_tr/3.;
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU] - dnoise_tr/3.;
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
 

  int vecino, particle;
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



  

  WxZ[i].x = wx + dtGPU * (fx / (volumeGPU*densfluidGPU)) - 1.5*advX + 0.5*advXGPU[i];
  WyZ[i].x = wy + dtGPU * (fy / (volumeGPU*densfluidGPU)) - 1.5*advY + 0.5*advYGPU[i];
  WzZ[i].x = wz + dtGPU * (fz / (volumeGPU*densfluidGPU)) - 1.5*advZ + 0.5*advZGPU[i];

  //WxZ[i].x = wx + dtGPU * (fx / (volumeGPU*densfluidGPU)) - 1.*advX ;
  //WyZ[i].x = wy + dtGPU * (fy / (volumeGPU*densfluidGPU)) - 1.*advY ;
  //WzZ[i].x = wz + dtGPU * (fz / (volumeGPU*densfluidGPU)) - 1.*advZ ;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

  //Save advection for the next time step
  advXGPU[i] = advX ;
  advYGPU[i] = advY ;
  advZGPU[i] = advZ ;

}

























__global__ void kernelConstructWQuasiNeutrallyBuoyantSemiImplicit(cufftDoubleComplex* WxZ,
								  cufftDoubleComplex* WyZ,
								  cufftDoubleComplex* WzZ,
								  double* fxboundaryGPU,
								  double* fyboundaryGPU,
								  double* fzboundaryGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  
  int vecino, particle;
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



  WxZ[i].x = -fx ;
  WyZ[i].x = -fy ;
  WzZ[i].x = -fz ;
  
  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}
