// Filename: calculateAdvectionFluid.cu
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


__global__ void calculateAdvectionFluid(const double* vxPredictionGPU,
					const double* vyPredictionGPU,
					const double* vzPredictionGPU,
					cufftDoubleComplex* vxZ,
					cufftDoubleComplex* vyZ,
					cufftDoubleComplex* vzZ){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  //for(int i=0;i<ncellsGPU;i++){
  
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  int vecinopxmy, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopymz, vecinomypz;

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

  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;
  
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vy0 = fetch_double(texVyGPU,vecino0);
  vz0 = fetch_double(texVzGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vy1 = fetch_double(texVyGPU,vecino1);
  vz1 = fetch_double(texVzGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vy2 = fetch_double(texVyGPU,vecino2);
  vz2 = fetch_double(texVzGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vy3 = fetch_double(texVyGPU,vecino3);
  vz3 = fetch_double(texVzGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vy4 = fetch_double(texVyGPU,vecino4);
  vz4 = fetch_double(texVzGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz5 = fetch_double(texVzGPU,vecino5);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);

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
  

  //24-10-2012, we should not add the particle kinetic stress twice!
  //vxPredictionGPU[i] = advX;
  //vyPredictionGPU[i] = advY;
  //vzPredictionGPU[i] = advZ;
  vxZ[i].x = advX;
  vyZ[i].x = advY;
  vzZ[i].x = advZ;
  //}


}
