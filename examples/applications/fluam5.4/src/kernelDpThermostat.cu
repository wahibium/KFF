// Filename: kernelDpThermostat.cu
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


__global__ void kernelDpThermostat_1(double* densityGPU, 
				     double* densityGPU2,
				     double* vxGPU, 
				     double* vyGPU,
				     double* vzGPU,
				     double* dmGPU,
				     double* dpxGPU, 
				     double* dpyGPU, 
				     double* dpzGPU,
				     double* d_rand, 
				     int substep,
				     double RK1, 
				     double RK2, 
				     double RK3){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

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

  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecinopxpx = tex1Dfetch(texvecino3GPU, vecino3);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecinopypy = tex1Dfetch(texvecino4GPU, vecino4);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopzpz = tex1Dfetch(texvecino5GPU, vecino5);

  vecinopxpy = tex1Dfetch(texvecinopxpyGPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopypz = tex1Dfetch(texvecinopypzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);



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
  double density3, density4, density5;

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
  
  sXX = pressure3 - pressure;// + 5.75242795456e-06 ;
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

  //Stress noise contribution
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  int n0;
  double fact1, fact2, fact4;
  fact1 = fact1GPU;
  fact2 = fact2GPU;
  fact4 = fact4GPU;

  n0 = substep * ncellsGPU * 12;
  
  fact1 = fact1GPU;
  fact2 = fact2GPU;
  fact4 = fact4GPU;
  
  dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellsGPU] + d_rand[n0 + vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
  sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellsGPU] + d_rand[n0 + vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellsGPU] + d_rand[n0 + vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + i + ncellsGPU];
  sXY += fact4 * dnoise_sXY;
  sYX += fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + i + 2*ncellsGPU];
  sXZ += fact4 * dnoise_sXZ;
  sZX += fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + i + 4*ncellsGPU];
  sYZ += fact4 * dnoise_sYZ;
  sZY += fact4 * dnoise_sYZ;
  
  dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellsGPU] + d_rand[n0 + i + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
  sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_sYY = d_rand[n0 + i + 3*ncellsGPU] - dnoise_tr/3.;
  sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_sZZ = d_rand[n0 + i + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + vecino1 + ncellsGPU];
  sXY -= fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellsGPU];
  sXZ -= fact4 * dnoise_sXZ;
  
  dnoise_sXY = d_rand[n0 + vecino2 + ncellsGPU];
  sYX -= fact4 * dnoise_sXY;
  
  dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellsGPU];
  sYZ -= fact4 * dnoise_sYZ;
  
  dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellsGPU];
  sZX -= fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellsGPU];
  sZY -= fact4 * dnoise_sYZ;
  
  if(RK3 !=0 ){

    n0 += ncellsGPU * 6;
    fact1 = RK3 * fact1GPU;
    fact2 = RK3 * fact2GPU;
    fact4 = RK3 * fact4GPU;
    dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellsGPU] + d_rand[n0 + vecino3 + 5*ncellsGPU];
    dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
    sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;

    dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellsGPU] + d_rand[n0 + vecino4 + 5*ncellsGPU];
    dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
    sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;

    dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellsGPU] + d_rand[n0 + vecino5 + 5*ncellsGPU];
    dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
    sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;

    dnoise_sXY = d_rand[n0 + i + ncellsGPU];
    sXY += fact4 * dnoise_sXY;
    sYX += fact4 * dnoise_sXY;

    dnoise_sXZ = d_rand[n0 + i + 2*ncellsGPU];
    sXZ += fact4 * dnoise_sXZ;
    sZX += fact4 * dnoise_sXZ;

    dnoise_sYZ = d_rand[n0 + i + 4*ncellsGPU];
    sYZ += fact4 * dnoise_sYZ;
    sZY += fact4 * dnoise_sYZ;

    dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellsGPU] + d_rand[n0 + i + 5*ncellsGPU];
    dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
    sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;

    dnoise_sYY = d_rand[n0 + i + 3*ncellsGPU] - dnoise_tr/3.;
    sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;

    dnoise_sZZ = d_rand[n0 + i + 5*ncellsGPU] - dnoise_tr/3.;
    sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;

    dnoise_sXY = d_rand[n0 + vecino1 + ncellsGPU];
    sXY -= fact4 * dnoise_sXY;

    dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellsGPU];
    sXZ -= fact4 * dnoise_sXZ;

    dnoise_sXY = d_rand[n0 + vecino2 + ncellsGPU];
    sYX -= fact4 * dnoise_sXY;

    dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellsGPU];
    sYZ -= fact4 * dnoise_sYZ;

    dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellsGPU];
    sZX -= fact4 * dnoise_sXZ;

    dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellsGPU];
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

}

__global__ void kernelDpThermostat_2(double* densityGPU,
				     double* vxGPU, 
				     double* vyGPU,
				     double* vzGPU,
				     double* dmGPU,
				     double* dpxGPU, 
				     double* dpyGPU, 
				     double* dpzGPU){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellstGPU) return;   
  
  int vecino3, vecino4, vecino5;
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);

  densityGPU[i] = dmGPU[i];

  vxGPU[i] = dpxGPU[i] * 2. / (dmGPU[i] + dmGPU[vecino3]);
  vyGPU[i] = dpyGPU[i] * 2. / (dmGPU[i] + dmGPU[vecino4]);
  vzGPU[i] = dpzGPU[i] * 2. / (dmGPU[i] + dmGPU[vecino5]);



}
