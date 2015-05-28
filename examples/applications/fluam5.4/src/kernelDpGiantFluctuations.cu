// Filename: kernelDpGiantFluctuations.cu
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


__global__ void kernelDpGiantFluctuations_1(double* densityGPU, 
					    double* densityGPU2,
					    double* vxGPU, 
					    double* vyGPU,
					    double* vzGPU,
					    double* dmGPU,
					    double* dpxGPU, 
					    double* dpyGPU, 
					    double* dpzGPU,
					    double* cGPU, 
					    double* cGPU2, 
					    double* dcGPU,
					    double* d_rand,
					    int* ghostIndex, 
					    int* realIndex,
					    int substep, 
					    double RK1, 
					    double RK2, 
					    double RK3){
  
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   
  int i = ghostIndex[j];

  double dm;
  double pressure, pressure3, pressure4, pressure5;
  double vx, vy, vz;

  double density = densityGPU[i];
  pressure = pressure_GPU(density);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 

  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopypz, vecinopymz, vecinomypz;
  int vecinopxpx, vecinopypy, vecinopzpz;

  vecino0 = i - mxmytGPU;
  vecino1 = i - mxtGPU;
  vecino2 = i - 1;
  vecino3 = i + 1;
  vecinopxpx = ghostIndex[ realIndex[vecino3] ] + 1;
  vecino4 = i + mxtGPU;
  vecinopypy = ghostIndex[ realIndex[vecino4] ] + mxtGPU;
  vecino5 = i + mxmytGPU;
  vecinopzpz = ghostIndex[ realIndex[vecino5] ] + mxmytGPU;
  
  vecinopxpy = i + 1 + mxtGPU;
  vecinopxmy = i + 1 - mxtGPU;
  vecinopxpz = i + 1 + mxmytGPU;
  vecinopxmz = i + 1 - mxmytGPU;
  vecinomxpy = i - 1 + mxtGPU;
  vecinomxpz = i - 1 + mxmytGPU;
  vecinopypz = i + mxtGPU + mxmytGPU;
  vecinopymz = i + mxtGPU - mxmytGPU;
  vecinomypz = i - mxtGPU + mxmytGPU;


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

  double density0, density1, density2;
  volatile double density3, density4, density5;

  density0 = densityGPU[vecino0];
  density1 = densityGPU[vecino1];
  density2 = densityGPU[vecino2];
  density3 = densityGPU[vecino3];
  pressure3 = pressure_GPU(density3);
  density4 = densityGPU[vecino4];
  pressure4 = pressure_GPU(density4);
  density5 = densityGPU[vecino5];
  pressure5 = pressure_GPU(density5);


  dm = 
    invdxGPU * ((density3 + density) * vx - (density + density2) * vx2) +
    invdyGPU * ((density4 + density) * vy - (density + density1) * vy1) +
    invdzGPU * ((density5 + density) * vz - (density + density0) * vz0);

  dm = -0.5 * dm;

  double sXX, sXY, sXZ;
  double sYX, sYY, sYZ;
  double sZX, sZY, sZZ;

  sXX = pressure3 - pressure;
  sYY = pressure4 - pressure;
  sZZ = pressure5 - pressure;

  sXX += 0.125 * ((density3+density)*vx + (densityGPU[vecinopxpx]+density3)*vx3)*(vx+vx3);
  sXX -= 0.125 * ((density+density2)*vx2 + (density3+density)*vx)*(vx2+vx);

  sYY += 0.125 * ((density4+density)*vy + (densityGPU[vecinopypy]+density4)*vy4)*(vy+vy4);
  sYY -= 0.125 * ((density+density1)*vy1 + (density4+density)*vy)*(vy1+vy);

  sZZ += 0.125 * ((density5+density)*vz + (densityGPU[vecinopzpz]+density5)*vz5)*(vz+vz5);
  sZZ -= 0.125 * ((density+density0)*vz0 + (density5+density)*vz)*(vz0+vz);

  double densitypxpy, densitypxmy, densitypxpz, densitypxmz;
  double densitymxpy, densitymxpz;
  double densitypypz, densitypymz, densitymypz;

  densitypxpy = densityGPU[vecinopxpy];
  densitypxmy = densityGPU[vecinopxmy];
  densitypxpz = densityGPU[vecinopxpz];
  densitypxmz = densityGPU[vecinopxmz];
  densitymxpy = densityGPU[vecinomxpy];
  densitymxpz = densityGPU[vecinomxpz];
  densitypypz = densityGPU[vecinopypz];
  densitypymz = densityGPU[vecinopymz];
  densitymypz = densityGPU[vecinomypz];

  
  sXY  = 0.125 * ((density4 + density)  * vy  + (densitypxpy + density3)    * vy3)    * (vx  + vx4);
  sXY -= 0.125 * ((density  + density1) * vy1 + (density3    + densitypxmy) * vypxmy) * (vx1 + vx);
  sXZ  = 0.125 * ((density5 + density)  * vz  + (densitypxpz + density3)    * vz3)    * (vx  + vx5);
  sXZ -= 0.125 * ((density  + density0) * vz0 + (density3    + densitypxmz) * vzpxmz) * (vx0 + vx);
  sYX  = 0.125 * ((density3 + density)  * vx  + (densitypxpy + density4)    * vx4)    * (vy  + vy3);
  sYX -= 0.125 * ((density  + density2) * vx2 + (density4    + densitymxpy) * vxmxpy) * (vy2 + vy);
  sYZ  = 0.125 * ((density5 + density)  * vz  + (densitypypz + density4)    * vz4)    * (vy  + vy5);
  sYZ -= 0.125 * ((density  + density0) * vz0 + (density4    + densitypymz) * vzpymz) * (vy0 + vy);
  sZX  = 0.125 * ((density3 + density)  * vx  + (densitypxpz + density5)    * vx5)    * (vz  + vz3);
  sZX -= 0.125 * ((density  + density2) * vx2 + (density5    + densitymxpz) * vxmxpz) * (vz2 + vz);
  sZY  = 0.125 * ((density4 + density)  * vy  + (densitypypz + density5)    * vy5)    * (vz  + vz4);
  sZY -= 0.125 * ((density  + density1) * vy1 + (density5    + densitymypz) * vymypz) * (vz1 + vz);

  //STRESS TENSOR CONTRIBUTION
  sXX -= 2. * shearviscosityGPU * invdxGPU * (vx3 - vx) + fact3GPU *
    (invdxGPU * (vx3 - vx) + invdyGPU * (vy3 - vypxmy) + invdzGPU * (vz3 - vzpxmz));
  sXX += 2. * shearviscosityGPU * invdxGPU * (vx - vx2) + fact3GPU *
    (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0));
  sYY -= 2. * shearviscosityGPU * invdyGPU * (vy4 - vy) + fact3GPU * 
    (invdxGPU * (vx4 - vxmxpy) + invdyGPU * (vy4 - vy) + invdzGPU * (vz4 - vzpymz));
  sYY += 2. * shearviscosityGPU * invdyGPU * (vy - vy1) + fact3GPU *
    (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0));
  sZZ -= 2. * shearviscosityGPU * invdzGPU * (vz5 - vz) + fact3GPU *
    (invdxGPU * (vx5 - vxmxpz) + invdyGPU * (vy5 - vymypz) + invdzGPU * (vz5 -vz));
  sZZ += 2. * shearviscosityGPU * invdzGPU * (vz - vz0) + fact3GPU *
    (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0));

  sXY -= shearviscosityGPU * (invdyGPU * (vx4 - vx) + invdxGPU * (vy3 - vy));
  sXY += shearviscosityGPU * (invdyGPU * (vx - vx1) + invdxGPU * (vypxmy - vy1));
  sXZ -= shearviscosityGPU * (invdzGPU * (vx5 - vx) + invdxGPU * (vz3 - vz));
  sXZ += shearviscosityGPU * (invdzGPU * (vx - vx0) + invdxGPU * (vzpxmz - vz0));
  sYX -= shearviscosityGPU * (invdxGPU * (vy3 - vy) + invdyGPU * (vx4 - vx));
  sYX += shearviscosityGPU * (invdxGPU * (vy - vy2) + invdyGPU * (vxmxpy - vx2));
  sYZ -= shearviscosityGPU * (invdzGPU * (vy5 - vy) + invdyGPU * (vz4 - vz));
  sYZ += shearviscosityGPU * (invdzGPU * (vy - vy0) + invdyGPU * (vzpymz - vz0));
  sZX -= shearviscosityGPU * (invdxGPU * (vz3 - vz) + invdzGPU * (vx5 - vx));
  sZX += shearviscosityGPU * (invdxGPU * (vz - vz2) + invdzGPU * (vxmxpz - vx2));
  sZY -= shearviscosityGPU * (invdyGPU * (vz4 - vz) + invdzGPU * (vy5 - vy));
  sZY += shearviscosityGPU * (invdyGPU * (vz - vz1) + invdzGPU * (vymypz - vy1));

  //Noise contribution
  int n0;
  n0 = substep * ncellstGPU * 18;
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  double fact1, fact2, fact4;
  fact1 = fact1GPU;
  fact2 = fact2GPU;
  fact4 = fact4GPU;
  
  dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellstGPU] + d_rand[n0 + vecino3 + 5*ncellstGPU];
  dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
  sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellstGPU] + d_rand[n0 + vecino4 + 5*ncellstGPU];
  dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellstGPU] - dnoise_tr/3.;
  sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellstGPU] + d_rand[n0 + vecino5 + 5*ncellstGPU];
  dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellstGPU] - dnoise_tr/3.;
  sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + i + ncellstGPU];
  sXY += fact4 * dnoise_sXY;
  sYX += fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + i + 2*ncellstGPU];
  sXZ += fact4 * dnoise_sXZ;
  sZX += fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + i + 4*ncellstGPU];
  sYZ += fact4 * dnoise_sYZ;
  sZY += fact4 * dnoise_sYZ;
  
  dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellstGPU] + d_rand[n0 + i + 5*ncellstGPU];
  dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
  sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_sYY = d_rand[n0 + i + 3*ncellstGPU] - dnoise_tr/3.;
  sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_sZZ = d_rand[n0 + i + 5*ncellstGPU] - dnoise_tr/3.;
  sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + vecino1 + ncellstGPU];
  sXY -= fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellstGPU];
  sXZ -= fact4 * dnoise_sXZ;
  
  dnoise_sXY = d_rand[n0 + vecino2 + ncellstGPU];
  sYX -= fact4 * dnoise_sXY;
  
  dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellstGPU];
  sYZ -= fact4 * dnoise_sYZ;
  
  dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellstGPU];
  sZX -= fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellstGPU];
  sZY -= fact4 * dnoise_sYZ;
  
  
  if(RK3 != 0){
    n0 += ncellstGPU * 9;
    fact1 = RK3 * fact1GPU;
    fact2 = RK3 * fact2GPU;
    fact4 = RK3 * fact4GPU;
    dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellstGPU] + d_rand[n0 + vecino3 + 5*ncellstGPU];
    dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
    sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;
    
    dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellstGPU] + d_rand[n0 + vecino4 + 5*ncellstGPU];
    dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellstGPU] - dnoise_tr/3.;
    sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;
    
    dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellstGPU] + d_rand[n0 + vecino5 + 5*ncellstGPU];
    dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellstGPU] - dnoise_tr/3.;
    sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;
    
    dnoise_sXY = d_rand[n0 + i + ncellstGPU];
    sXY += fact4 * dnoise_sXY;
    sYX += fact4 * dnoise_sXY;
    
    dnoise_sXZ = d_rand[n0 + i + 2*ncellstGPU];
    sXZ += fact4 * dnoise_sXZ;
    sZX += fact4 * dnoise_sXZ;
    
    dnoise_sYZ = d_rand[n0 + i + 4*ncellstGPU];
    sYZ += fact4 * dnoise_sYZ;
    sZY += fact4 * dnoise_sYZ;
    
    dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellstGPU] + d_rand[n0 + i + 5*ncellstGPU];
    dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
    sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;
    
    dnoise_sYY = d_rand[n0 + i + 3*ncellstGPU] - dnoise_tr/3.;
    sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;
    
    dnoise_sZZ = d_rand[n0 + i + 5*ncellstGPU] - dnoise_tr/3.;
    sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;
    
    dnoise_sXY = d_rand[n0 + vecino1 + ncellstGPU];
    sXY -= fact4 * dnoise_sXY;
    
    dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellstGPU];
    sXZ -= fact4 * dnoise_sXZ;
    
    dnoise_sXY = d_rand[n0 + vecino2 + ncellstGPU];
    sYX -= fact4 * dnoise_sXY;
    
    dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellstGPU];
    sYZ -= fact4 * dnoise_sYZ;
    
    dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellstGPU];
    sZX -= fact4 * dnoise_sXZ;
    
    dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellstGPU];
    sZY -= fact4 * dnoise_sYZ;
  }
  
  
  double px = vxGPU[i] * 0.5 * (densityGPU2[i] + densityGPU2[vecino3]) * RK1;
  double py = vyGPU[i] * 0.5 * (densityGPU2[i] + densityGPU2[vecino4]) * RK1;
  double pz = vzGPU[i] * 0.5 * (densityGPU2[i] + densityGPU2[vecino5]) * RK1;

  px += vx * 0.5 * (density + density3) * RK2;
  py += vy * 0.5 * (density + density4) * RK2;
  pz += vz * 0.5 * (density + density5) * RK2;

  
  
  px += -(invdxGPU * sXX + invdyGPU * sXY + invdzGPU * sXZ)*dtGPU*RK2;
  py += -(invdxGPU * sYX + invdyGPU * sYY + invdzGPU * sYZ)*dtGPU*RK2;
  pz += -(invdxGPU * sZX + invdyGPU * sZY + invdzGPU * sZZ)*dtGPU*RK2;

  dmGPU[i] = RK1 * densityGPU2[i] + RK2 * (density + dm * dtGPU);
  
  dpxGPU[i] = px;
  dpyGPU[i] = py;
  dpzGPU[i] = pz;




  //Concentration
  //Concentration
  double dc;
  //double pressure2, pressure1, pressure0;
  //pressure0 = pressure_GPU(density0);
  //pressure1 = pressure_GPU(density1);
  //pressure2 = pressure_GPU(density2);
  dc = 0;
  
  dc = invdxGPU * ((cGPU[vecino3]*density3 + cGPU[i]*density) * vx
		   -(cGPU[i]*density + cGPU[vecino2]*density2) * vx2) +
    invdyGPU * ((cGPU[vecino4]*density4 + cGPU[i]*density) * vy
		-(cGPU[i]*density + cGPU[vecino1]*density1) * vy1) +
    invdzGPU * ((cGPU[vecino5]*density5 + cGPU[i]*density) * vz
		-(cGPU[i]*density + cGPU[vecino0]*density0) * vz0);
  
  dc -= invdxGPU*invdxGPU*((funcDiffusionBM(vecino3) + funcDiffusionBM(i))*
			   (cGPU[vecino3] - cGPU[i]) -
			   (funcDiffusionBM(i) + funcDiffusionBM(vecino2))*
			   (cGPU[i] - cGPU[vecino2])) +
    invdyGPU*invdyGPU*((funcDiffusionBM(vecino4) + funcDiffusionBM(i))*
		       (cGPU[vecino4] - cGPU[i]) -
		       (funcDiffusionBM(i) + funcDiffusionBM(vecino1))*
		       (cGPU[i] - cGPU[vecino1])) +
    invdzGPU*invdzGPU*((funcDiffusionBM(vecino5) + funcDiffusionBM(i))*
		       (cGPU[vecino5] - cGPU[i]) -
		       (funcDiffusionBM(i) + funcDiffusionBM(vecino0))*
		       (cGPU[i] - cGPU[vecino0]));
  
  dc -= invdyGPU * funcDiffusionBM(i) * ((cGPU[vecino4]+cGPU[i])-(cGPU[i]+cGPU[vecino1])) *
    soretCoefficientGPU * gradTemperatureGPU;
  
  dc = -0.5 * dc;

  /*
  dc += fact5GPU * (
    invdxGPU*invdxGPU*((baroDiffusion(pressure3,cGPU[vecino3]) + baroDiffusion(pressure,cGPU[i]))*
			   (pressure3 - pressure) -
 			   (baroDiffusion(pressure,cGPU[i]) + baroDiffusion(pressure2,cGPU[vecino2]))*
			   (pressure - pressure2)) +
    invdyGPU*invdyGPU*((baroDiffusion(pressure4,cGPU[vecino4]) + baroDiffusion(pressure,cGPU[i]))*
		       (pressure4 - pressure) -
		       (baroDiffusion(pressure,cGPU[i]) + baroDiffusion(pressure1,cGPU[vecino1]))*
		       (pressure - pressure1)) +
    invdzGPU*invdzGPU*((baroDiffusion(pressure5,cGPU[vecino5]) + baroDiffusion(pressure,cGPU[i]))*
		       (pressure5 - pressure) -
		       (baroDiffusion(pressure,cGPU[i]) + baroDiffusion(pressure0,cGPU[vecino0]))*
		       (pressure - pressure0)));
  */
  //Noise terms
  n0 = substep * ncellstGPU * 18;
  double c = 0.5 * (cGPU[vecino3] + cGPU[i]);
  dc += fact5GPU*invdxGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + i + 6*ncellstGPU];
  c = 0.5 * (cGPU[i] + cGPU[vecino2]);
  dc -= fact5GPU*invdxGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + vecino2 + 6*ncellstGPU];

  c = 0.5 * (cGPU[i] + cGPU[vecino4]);
  dc += fact5GPU*invdyGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + i + 7*ncellstGPU];
  c = 0.5 * (cGPU[i] + cGPU[vecino1]);
  dc -= fact5GPU*invdyGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + vecino1 + 7*ncellstGPU];

  c = 0.5 * (cGPU[i] + cGPU[vecino5]);
  dc += fact5GPU*invdzGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + i + 8*ncellstGPU];
  c = 0.5 * (cGPU[i] + cGPU[vecino0]);
  dc -= fact5GPU*invdzGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + vecino0 + 8*ncellstGPU];

  if(RK3 != 0){
    n0 += ncellstGPU * 9;
    c = 0.5 * (cGPU[vecino3] + cGPU[i]);
    dc += fact5GPU*RK3*invdxGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c)+massSpecies0GPU*c))*d_rand[n0 + i + 6*ncellstGPU];
    c = 0.5 * (cGPU[i] + cGPU[vecino2]);
    dc -= fact5GPU*RK3*invdxGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c)+massSpecies0GPU*c))*d_rand[n0 + vecino2 + 6*ncellstGPU];
    
    c = 0.5 * (cGPU[i] + cGPU[vecino4]);
    dc += fact5GPU*RK3*invdyGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + i + 7*ncellstGPU];
    c = 0.5 * (cGPU[i] + cGPU[vecino1]);
    dc -= fact5GPU*RK3*invdyGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + vecino1 + 7*ncellstGPU];
    
    c = 0.5 * (cGPU[i] + cGPU[vecino5]);
    dc += fact5GPU*RK3*invdzGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + i + 8*ncellstGPU];
    c = 0.5 * (cGPU[i] + cGPU[vecino0]);
    dc -= fact5GPU*RK3*invdzGPU * sqrt(2*funcDiffusionBM(i)*c*(1-c)*(massSpecies1GPU*(1-c) + massSpecies0GPU*c))*d_rand[n0 + vecino0 + 8*ncellstGPU];
  }
  
  
  dcGPU[i] = RK1 * densityGPU2[i] * cGPU2[i] + RK2 * (density*cGPU[i] + dc * dtGPU);


  
  

















}

__global__ void kernelDpGiantFluctuations_2(double* densityGPU,
					    double* vxGPU, 
					    double* vyGPU,
					    double* vzGPU,
					    double* dmGPU,
					    double* dpxGPU, 
					    double* dpyGPU, 
					    double* dpzGPU,
					    double* cGPU, 
					    double* dcGPU,
					    int* ghostIndex, 
					    int* realIndex){
  

  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   
  int i = ghostIndex[j];
  
  int vecino3, vecino4, vecino5;
  vecino3 = ghostIndex[ realIndex[i + 1] ];
  vecino4 = ghostIndex[ realIndex[i + mxtGPU ] ];
  vecino5 = ghostIndex[ realIndex[i + mxmytGPU] ];


  densityGPU[i] = dmGPU[i];


  vxGPU[i] = dpxGPU[i] * 2. / (dmGPU[i] + dmGPU[vecino3]);
  vyGPU[i] = dpyGPU[i] * 2. / (dmGPU[i] + dmGPU[vecino4]);
  vzGPU[i] = dpzGPU[i] * 2. / (dmGPU[i] + dmGPU[vecino5]);

  cGPU[i] = dcGPU[i] / dmGPU[i];

}
