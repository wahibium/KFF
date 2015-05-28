//STEP 2: (1-(omega1*dt*c)^2 * L) * rho^* = W
//Construct vector W
__global__ void kernelConstructWSemiImplicitCompressibleParticles_1(const double *densityGPU,
								    const double *dpxGPU,
								    const double *dpyGPU,
								    const double *dpzGPU,
								    cufftDoubleComplex *vxZ,
								    const double omega1,
								    const double omega2){
								    

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   


  //double omega1 = 1.70710678118654752;
  //double omega1 = 0.292893218813452476;
  //double omega2 = 0.5;

  double density = densityGPU[i];

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);

  double vx, vy, vz, vx2, vy1, vz0;
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vz0 = fetch_double(texVzGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);

  double density0, density1, density2, density3, density4, density5;
  density0 = densityGPU[vecino0];
  density1 = densityGPU[vecino1];
  density2 = densityGPU[vecino2];
  density3 = densityGPU[vecino3];
  density4 = densityGPU[vecino4];
  density5 = densityGPU[vecino5];


  double W = density ;

  //Contrinution momentum g^n
  W -= (omega2-omega1) * dtGPU * 0.5 * (invdxGPU * ((density3 + density) * vx - (density + density2) * vx2) +
	      invdyGPU * ((density4 + density) * vy - (density + density1) * vy1) +
	      invdzGPU * ((density5 + density) * vz - (density + density0) * vz0) );

  //Contribution from the mometum equation
  W -= omega1 * dtGPU * (invdxGPU * (dpxGPU[i]-dpxGPU[vecino2]) +
			 invdyGPU * (dpyGPU[i]-dpyGPU[vecino1]) +
			 invdzGPU * (dpzGPU[i]-dpzGPU[vecino0]));


  vxZ[i].x = W;
  vxZ[i].y = 0;


  


}





















//STEP 6: (1-(omega4*dt*c)^2 * L) * rho^{n+1} = W
//Construct vector W
__global__ void kernelConstructWSemiImplicitCompressibleParticles_2(const double *densityGPU,
								    const double *vxPredictionGPU,
								    const double *vyPredictionGPU,
								    const double *vzPredictionGPU,
								    const double *dpxGPU,
								    const double *dpyGPU,
								    const double *dpzGPU,
								    const cufftDoubleComplex *vxZ,
								    cufftDoubleComplex *vyZ,
								    const double omega3,
								    const double omega4){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega3 = -2.41421356237309505;
  //double omega3 = -0.414213562373095049;
  //double omega4 = 1.70710678118654752;
  //double omega4 = 0.292893218813452476;


  double density = densityGPU[i];

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);

  double vx, vy, vz, vx2, vy1, vz0;
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vz0 = fetch_double(texVzGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);

  double density0, density1, density2, density3, density4, density5;
  density0 = densityGPU[vecino0];
  density1 = densityGPU[vecino1];
  density2 = densityGPU[vecino2];
  density3 = densityGPU[vecino3];
  density4 = densityGPU[vecino4];
  density5 = densityGPU[vecino5];


  
  double W = density ;

  //Contrinution momentum g^n
  W -= (1-omega3-omega4) * dtGPU * 0.5 * 
    (invdxGPU * ((density3 + density) * vx - (density + density2) * vx2) +
     invdyGPU * ((density4 + density) * vy - (density + density1) * vy1) +
     invdzGPU * ((density5 + density) * vz - (density + density0) * vz0) );


  //Contribution at time n+omega2
  density = vxZ[i].x / ncellsGPU;

  vx = vxPredictionGPU[i];
  vy = vyPredictionGPU[i];
  vz = vzPredictionGPU[i];
  vz0 = vzPredictionGPU[vecino0];
  vy1 = vyPredictionGPU[vecino1];
  vx2 = vxPredictionGPU[vecino2];

  density0 = vxZ[vecino0].x / ncellsGPU;
  density1 = vxZ[vecino1].x / ncellsGPU;
  density2 = vxZ[vecino2].x / ncellsGPU;
  density3 = vxZ[vecino3].x / ncellsGPU;
  density4 = vxZ[vecino4].x / ncellsGPU;
  density5 = vxZ[vecino5].x / ncellsGPU;


  //Contrinution momentum g^*
  W -= omega3 * dtGPU * 0.5 * (invdxGPU * ((density3 + density) * vx - (density + density2) * vx2) +
	      invdyGPU * ((density4 + density) * vy - (density + density1) * vy1) +
	      invdzGPU * ((density5 + density) * vz - (density + density0) * vz0) );


  //Contribution from the mometum equation
  W -= omega4 * dtGPU * (invdxGPU * (dpxGPU[i]-dpxGPU[vecino2]) +
			 invdyGPU * (dpyGPU[i]-dpyGPU[vecino1]) +
			 invdzGPU * (dpzGPU[i]-dpzGPU[vecino0]));


  vyZ[i].x = W;
  vyZ[i].y = 0;



}
