// Filename: runSchemeCompressibleParticles.cu
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


bool runSchemeSemiImplicitCompressibleParticles(){
  //L-stable
  //double omega1 = 1.70710678118654752;
  //double omega2 = 0.5;
  //double omega3 = -2.41421356237309505;
  //double omega4 = 1.70710678118654752;
  //double omega5 = 1;

  //L-stable
  //double omega1 = 0.292893218813452476;
  //double omega2 = 0.5;
  //double omega3 = -0.414213562373095049;
  //double omega4 = 0.292893218813452476;
  //double omega5 = 1;

  //Mid point
  double omega1 = 0.5;
  double omega2 = 0.5;
  double omega3 = 0;
  double omega4 = 0.5;
  double omega5 = 1;

  //Trapezoidal
  /*double omega1 = 0.5;
  double omega2 = 1;
  double omega3 = 0;
  double omega4 = 0.5;
  double omega5 = 0.5;*/


  /*double omega1 = 0.5;
  double omega2 = 0.5;
  double omega3 = 0.666666666666666667;
  double omega4 = 0.166666666666666667;
  double omega5 = 1.;*/




  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  int threadsPerBlockBoundary = 128;
  if((nboundary/threadsPerBlockBoundary) < 60) threadsPerBlockBoundary = 64;
  if((nboundary/threadsPerBlockBoundary) < 60) threadsPerBlockBoundary = 32;
  int numBlocksBoundary = (nboundary-1)/threadsPerBlockBoundary + 1;

  int threadsPerBlockPartAndBoundary = 128;
  if(((np+nboundary)/threadsPerBlockPartAndBoundary) < 60) threadsPerBlockPartAndBoundary = 64;
  if(((np+nboundary)/threadsPerBlockPartAndBoundary) < 60) threadsPerBlockPartAndBoundary = 32;
  int numBlocksPartAndBoundary = (np+nboundary-1)/threadsPerBlockPartAndBoundary + 1;

  int threadsPerBlockParticles = 128;
  if((np/threadsPerBlockParticles) < 60) threadsPerBlockParticles = 64;
  if((np/threadsPerBlockParticles) < 60) threadsPerBlockParticles = 32;
  int numBlocksParticles = (np-1)/threadsPerBlockParticles + 1;

  int threadsPerBlockNeighbors, numBlocksNeighbors;
  if(ncells>numNeighbors){
    threadsPerBlockNeighbors = 128;
    if((ncells/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 64;
    if((ncells/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 32;
    numBlocksNeighbors = (ncells-1)/threadsPerBlockNeighbors + 1;
  }
  else{
    threadsPerBlockNeighbors = 128;
    if((numNeighbors/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 64;
    if((numNeighbors/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 32;
    numBlocksNeighbors = (numNeighbors-1)/threadsPerBlockNeighbors + 1;
  }














  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 12 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;

  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,
						   vecino2GPU,
						   vecino3GPU,
						   vecino4GPU,
						   vecinopxpyGPU,
						   vecinopxmyGPU,
						   vecinopxpzGPU,
						   vecinopxmzGPU,
						   vecinomxpyGPU,
						   vecinomxmyGPU,
						   vecinomxpzGPU,
						   vecinomxmzGPU,
						   vecinopypzGPU,
						   vecinopymzGPU,
						   vecinomypzGPU,
						   vecinomymzGPU,
						   vecinopxpypzGPU,
						   vecinopxpymzGPU,
						   vecinopxmypzGPU,
						   vecinopxmymzGPU,
						   vecinomxpypzGPU,
						   vecinomxpymzGPU,
						   vecinomxmypzGPU,
						   vecinomxmymzGPU);

  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,
						    vecino1GPU,
						    vecino2GPU,
						    vecino3GPU,
						    vecino4GPU,
						    vecino5GPU);



  //Initialize plan
  cufftHandle FFT;
  cufftPlan3d(&FFT,mz,my,mx,CUFFT_Z2Z);

  //Initialize factors for fourier space update
  int threadsPerBlockdim, numBlocksdim;
  if((mx>=my)&&(mx>=mz)){
    threadsPerBlockdim = 128;
    numBlocksdim = (mx-1)/threadsPerBlockdim + 1;
  }
  else if((my>=mz)){
    threadsPerBlockdim = 128;
    numBlocksdim = (my-1)/threadsPerBlockdim + 1;
  }
  else{
    threadsPerBlockdim = 128;
    numBlocksdim = (mz-1)/threadsPerBlockdim + 1;
  }
  initializePrefactorFourierSpace_1<<<1,1>>>(gradKx,
					     gradKy,
					     gradKz,
					     expKx,
					     expKy,
					     expKz,
					     pF);
  
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);



  while(step<numsteps){
    //Generate random numbers
    generateRandomNumbers(numberRandom);

    //Boundaries and particles 
    //Update particles position to q^{n+1/2}
    //Spread and interpolate particles force
    boundaryParticlesFunctionCompressibleParticles(0,
						   numBlocksBoundary,
						   threadsPerBlockBoundary,
						   numBlocksNeighbors,
						   threadsPerBlockNeighbors,
						   numBlocksPartAndBoundary,
						   threadsPerBlockPartAndBoundary,
						   numBlocksParticles,
						   threadsPerBlockParticles,
						   numBlocks,
						   threadsPerBlock);


    //STEP 1: g^* = g^n - omega1*dt*c^2*G*rho^* + A^n
    //Calculate A^n
    kernelDpSemiImplicitCompressibleParticles_1<<<numBlocks,threadsPerBlock>>>(densityGPU,
									       dpxGPU,
									       dpyGPU,
									       dpzGPU,
									       dRand,
									       fxboundaryGPU,
									       fyboundaryGPU,
									       fzboundaryGPU,
									       omega1,
									       omega2,
									       step);
  

    //STEP 2: (1-(omega1*dt*c)^2 * L) * rho^* = W
    //Construct vector W
    kernelConstructWSemiImplicitCompressibleParticles_1<<<numBlocks,threadsPerBlock>>>(densityGPU,
										       dpxGPU,
										       dpyGPU,
										       dpzGPU,
										       vxZ,
										       omega1,
										       omega2);

    //STEP 3: solve rho^* in fourier space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    kernelUpdateRhoSemiImplicit<<<numBlocks,threadsPerBlock>>>(vxZ,pF,omega1);//W
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);

    //STEP 4: solve vx^*
    kernelDpSemiImplicitCompressibleParticles_2<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
									       vyPredictionGPU,
									       vzPredictionGPU,
									       dpxGPU,
									       dpyGPU,
									       dpzGPU,
									       vxZ,
									       omega1);

    //STEP 5: g^{n+1} = g^n - omega1*dt*c^2*G*rho^{n+1} + A^{n+1/2}
    //Calculate A^{n+1/2}
    kernelDpSemiImplicitCompressibleParticles_3<<<numBlocks,threadsPerBlock>>>(densityGPU,
									       vxPredictionGPU,
									       vyPredictionGPU,
									       vzPredictionGPU,
									       dpxGPU,
									       dpyGPU,
									       dpzGPU,
									       dRand,
									       fxboundaryGPU,
									       fyboundaryGPU,
									       fzboundaryGPU,
									       vxZ,
									       omega2,
									       omega3,
									       omega4,
									       omega5,
									       step);//vxZ

    //STEP 6: (1-(omega4*dt*c)^2 * L) * rho^{n+1} = W
    //Construct vector W
    kernelConstructWSemiImplicitCompressibleParticles_2<<<numBlocks,threadsPerBlock>>>(densityGPU,
										       vxPredictionGPU,
										       vyPredictionGPU,
										       vzPredictionGPU,
										       dpxGPU,
										       dpyGPU,
										       dpzGPU,
										       vxZ,
										       vyZ,
										       omega3,
										       omega4);

    //STEP 7: solve rho^{n+1} in fourier space
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    kernelUpdateRhoSemiImplicit_2<<<numBlocks,threadsPerBlock>>>(vyZ,pF,omega4);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    
    
    //Copy v^n to vxPredictionGPU
    //We need it for the particle update
    copyField<<<numBlocks,threadsPerBlock>>>(vxGPU,
					     vyGPU,
					     vzGPU,
					     vxPredictionGPU,
					     vyPredictionGPU,
					     vzPredictionGPU);


    //STEP 8: solve vx^{n+1} and rho^{n+1}
    kernelDpSemiImplicitCompressibleParticles_4<<<numBlocks,threadsPerBlock>>>(vyZ,
									       densityGPU,
									       vxGPU,
									       vyGPU,
									       vzGPU,
									       dpxGPU,
									       dpyGPU,
									       dpzGPU,
									       omega4);    

    //Boundaries and particles part start
    boundaryParticlesFunctionCompressibleParticles(1,
						   numBlocksBoundary,
						   threadsPerBlockBoundary,
						   numBlocksNeighbors,
						   threadsPerBlockNeighbors,
						   numBlocksPartAndBoundary,
						   threadsPerBlockPartAndBoundary,
						   numBlocksParticles,
						   threadsPerBlockParticles,
						   numBlocks,
						   threadsPerBlock);
    
    step++;

    if(!(step%samplefreq)&&(step>0)){
      cout << "Semi-Implicit compressible  " << step << endl;
      if(!gpuToHostParticles()) return 0;
      if(!saveFunctionsSchemeSemiImplicitCompressibleParticles(1,step)) return 0;
    }
    
  }


  //Free FFT
  cufftDestroy(FFT);
  freeRandomNumbersGPU();




  return 1;
}

