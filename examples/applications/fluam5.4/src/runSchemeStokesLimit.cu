// Filename: runSchemeStokesLimit.cu
//
// Copyright (c) 2010-2012, Florencio Balboa Usabiaga
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


bool runSchemeStokesLimit(){
   

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
  size_t numberRandom = 6*ncells + 9*np;
  if(numberRandom%2){//This is a limitation of curand library, numberRandom should be even
    numberRandom++ ;
  }
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
    
    //STEP 1:
    //Boundaries and particles 
    //Spread particles force and drift
    boundaryParticlesFunctionStokesLimit(0,
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

    
    //STEP 2
    //Construct W=S*F + noise + thermalDrift
    kernelConstructWstokesLimit<<<numBlocks,threadsPerBlock>>>(vxGPU, //Stored fx=S*Fx+S*drift_p-S*drift_m
							       vyGPU,
							       vzGPU,
							       vxZ,
							       vyZ,
							       vzZ,
							       dRand);

    
    //STEP 3
    //Solve fluid velocity field
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    kernelUpdateVstokesLimit<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,
								   vyZ,
								   vzZ,
								   vxGPU,
								   vyGPU,
								   vzGPU);


    //STEP 4:
    //Boundaries and particles 
    //Update particle position with the midpoint scheme
    // q^{n+1/2} = q^n + 0.5*dt*extraMobility*F^n + 0.5*dt*J^n * v + noise_1
    // q^{n+1} = q^n + dt*extraMobility*F^{n+1/2} + dt*J^{n+1/2} * v + noise_1 + noise_2
    boundaryParticlesFunctionStokesLimit(1,
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
      cout << "Stokes Limit  " << step << endl;
      if(!gpuToHostStokesLimit()) return 0;
      if(!saveFunctionsSchemeStokesLimit(1,step)) return 0;
    }
    
  }


  //Free FFT
  cufftDestroy(FFT);
  freeRandomNumbersGPU();




  return 1;
}

