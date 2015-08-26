
// Filename: kernelConstructW.cu
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
//In the first substep
//  W = u^n + 0.5*dt*nu*L*u^n + Advection(u^n) + (dt/rho)*f^n_{noise}
//
//In the second substep
//  W = u^n + 0.5*dt*nu*L*u^n + Advection(u^{n+1/2}) + (dt/rho)*f^n_{noise}
//with u^{n+1/2} = 0.5 * (u^n + u^{n+1}_{result from first substep})


__global__ void kernelConstructW_1(double *vxPredictionGPU, 
				   double *vyPredictionGPU, 
				   double *vzPredictionGPU, 
				   cufftDoubleComplex *WxZ, 
				   cufftDoubleComplex *WyZ, 
				   cufftDoubleComplex *WzZ, 
				   double *d_rand){
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
  //vecinopxpx = tex1Dfetch(texvecino3GPU, vecino3);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  //vecinopypy = tex1Dfetch(texvecino4GPU, vecino4);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  //vecinopzpz = tex1Dfetch(texvecino5GPU, vecino5);
  //vecinopxpy = tex1Dfetch(texvecinopxpyGPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  //vecinopxpz = tex1Dfetch(texvecinopxpzGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  //vecinopypz = tex1Dfetch(texvecinopypzGPU,i);
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
  wx += vx;
  wy += vy;
  wz += vz;
  
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
  wx -= advX;
  wy -= advY;
  wz -= advZ;

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
  
  /*dnoise_sXX = d_rand[vecino3];
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[vecino4 + 4*ncellsGPU];
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[vecino5 + 8*ncellsGPU];
  wz += invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4GPU * dnoise_sXY;
  dnoise_sXY = d_rand[i + 3*ncellsGPU];
  wy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  wx += invdzGPU * fact4GPU * dnoise_sXZ;
  dnoise_sXZ = d_rand[i + 6*ncellsGPU];
  wz += invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 5*ncellsGPU];
  wy += invdzGPU * fact4GPU * dnoise_sYZ;
  dnoise_sYZ = d_rand[i + 7*ncellsGPU];
  wz += invdyGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXX = d_rand[i];
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 4*ncellsGPU];
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 8*ncellsGPU];
  wz -= invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  wx -= invdzGPU * fact4GPU * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + 3*ncellsGPU];
  wy -= invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 5*ncellsGPU];
  wy -= invdzGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 6*ncellsGPU];
  wz -= invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 7*ncellsGPU];
  wz -= invdyGPU * fact4GPU * dnoise_sYZ;*/

  WxZ[i].x = wx;
  WyZ[i].x = wy;
  WzZ[i].x = wz;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}






__global__ void kernelConstructW_2(double *vxPredictionGPU, 
				   double *vyPredictionGPU, 
				   double *vzPredictionGPU, 
				   cufftDoubleComplex *WxZ, 
				   cufftDoubleComplex *WyZ, 
				   cufftDoubleComplex *WzZ, 
				   double *d_rand){
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
  //vecinopxpx = tex1Dfetch(texvecino3GPU, vecino3);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  //vecinopypy = tex1Dfetch(texvecino4GPU, vecino4);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  //vecinopzpz = tex1Dfetch(texvecino5GPU, vecino5);
  //vecinopxpy = tex1Dfetch(texvecinopxpyGPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  //vecinopxpz = tex1Dfetch(texvecinopxpzGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  //vecinopypz = tex1Dfetch(texvecinopypzGPU,i);
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
  wx += vx;
  wy += vy;
  wz += vz;
  
  //Advection part
  double advX, advY, advZ;
  vx = vxPredictionGPU[i];
  vy = vyPredictionGPU[i];
  vz = vzPredictionGPU[i];
  vx0 = vxPredictionGPU[vecino0];
  vx1 = vxPredictionGPU[vecino1];
  vx2 = vxPredictionGPU[vecino2];
  vx3 = vxPredictionGPU[vecino3];
  vx4 = vxPredictionGPU[vecino4];
  vx5 = vxPredictionGPU[vecino5];
  vy0 = vyPredictionGPU[vecino0];
  vy1 = vyPredictionGPU[vecino1];
  vy2 = vyPredictionGPU[vecino2];
  vy3 = vyPredictionGPU[vecino3];
  vy4 = vyPredictionGPU[vecino4];
  vy5 = vyPredictionGPU[vecino5];
  vz0 = vzPredictionGPU[vecino0];
  vz1 = vzPredictionGPU[vecino1];
  vz2 = vzPredictionGPU[vecino2];
  vz3 = vzPredictionGPU[vecino3];
  vz4 = vzPredictionGPU[vecino4];
  vz5 = vzPredictionGPU[vecino5];
  vxmxpy = vxPredictionGPU[vecinomxpy];
  vxmxpz = vxPredictionGPU[vecinomxpz];
  vypxmy = vyPredictionGPU[vecinopxmy];
  vymypz = vyPredictionGPU[vecinomypz];
  vzpxmz = vzPredictionGPU[vecinopxmz];
  vzpymz = vzPredictionGPU[vecinopymz];
  
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
  wx -= advX;
  wy -= advY;
  wz -= advZ;

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
  
  /*dnoise_sXX = d_rand[vecino3];
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[vecino4 + 4*ncellsGPU];
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[vecino5 + 8*ncellsGPU];
  wz += invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4GPU * dnoise_sXY;
  dnoise_sXY = d_rand[i + 3*ncellsGPU];
  wy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  wx += invdzGPU * fact4GPU * dnoise_sXZ;
  dnoise_sXZ = d_rand[i + 6*ncellsGPU];
  wz += invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 5*ncellsGPU];
  wy += invdzGPU * fact4GPU * dnoise_sYZ;
  dnoise_sYZ = d_rand[i + 7*ncellsGPU];
  wz += invdyGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXX = d_rand[i];
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 4*ncellsGPU];
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 8*ncellsGPU];
  wz -= invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  wx -= invdzGPU * fact4GPU * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + 3*ncellsGPU];
  wy -= invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 5*ncellsGPU];
  wy -= invdzGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 6*ncellsGPU];
  wz -= invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 7*ncellsGPU];
  wz -= invdyGPU * fact4GPU * dnoise_sYZ;*/

  WxZ[i].x = wx;
  WyZ[i].x = wy;
  WzZ[i].x = wz;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}
