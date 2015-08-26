// Filename: runSchemeIncompressibleBinaryMixture.cu
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


bool runSchemeIncompressibleBinaryMixture(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 9 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;

  //Initialize neighbors list
  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,vecino2GPU,vecino3GPU,vecino4GPU,
						   vecinopxpyGPU,vecinopxmyGPU,vecinopxpzGPU,vecinopxmzGPU,
						   vecinomxpyGPU,vecinomxmyGPU,vecinomxpzGPU,vecinomxmzGPU,
						   vecinopypzGPU,vecinopymzGPU,vecinomypzGPU,vecinomymzGPU,
						   vecinopxpypzGPU,vecinopxpymzGPU,vecinopxmypzGPU,
						   vecinopxmymzGPU,
						   vecinomxpypzGPU,vecinomxpymzGPU,vecinomxmypzGPU,
						   vecinomxmymzGPU);
  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,vecino1GPU,vecino2GPU,
						    vecino3GPU,vecino4GPU,vecino5GPU);


  //Initialize plan
  cufftHandle FFT;
  cufftPlan3d(&FFT,mz,my,mx,CUFFT_Z2Z);
  
  //Initialize factors for update in fourier space
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
  initializePrefactorFourierSpace_1<<<1,1>>>(gradKx,gradKy,gradKz,expKx,expKy,expKz,pF);
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);

  // A. Donev: Project the initial velocity to make sure it is div-free
  //---------------------------------------------------------
  //Copy velocities to complex variable
  doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,vxZ,vyZ,vzZ);

  //Take velocities to fourier space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);

  //Project into divergence free space
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);

  //Take velocities to real space
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

  //Copy velocities to real variables
  doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);
  //---------------------------------------------------------
  
  while(step<numsteps){

    // Donev: Moved this here so the initial configuration is saved as well
    //if(!(step%samplefreq)&&(step>0)){
    if((step>=0)&&(!(step%abs(samplefreq)))&&(abs(samplefreq)>0)){
      cout << "INCOMPRESSIBLE UpdateHydroGrid" << step << endl;
      if(!gpuToHostIncompressible()) return 0;
      if(!saveFunctionsSchemeIncompressibleBinaryMixture(1)) return 0;
    }

    //Generate random numbers
    generateRandomNumbers(numberRandom);

    //First substep
    kernelConstructWBinaryMixture_1<<<numBlocks,threadsPerBlock>>>(vxGPU,
								   vyGPU,
								   vzGPU,
								   cGPU,
                                                                   cGPU,
								   vxZ,
								   vyZ,
								   vzZ,
								   cZ,
								   dRand);

    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_FORWARD);
    kernelShiftBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,-1);
    kernelUpdateIncompressibleBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF);
    kernelShiftBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_INVERSE);
    calculateVelocityPredictionBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,
									    vyZ,
									    vzZ,
									    cZ,
									    vxPredictionGPU,
									    vyPredictionGPU,
									    vzPredictionGPU,
									    cGPU,
									    cPredictionGPU);
     

    //Second substep
    kernelConstructWBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
								   vyPredictionGPU,
								   vzPredictionGPU,
								   cGPU,
								   cPredictionGPU,
								   vxZ,
								   vyZ,
								   vzZ,
								   cZ,
								   dRand);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_FORWARD);
    kernelShiftBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,-1);
    kernelUpdateIncompressibleBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF);
    kernelShiftBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalizedBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,
										vyZ,
										vzZ,
										cZ,
										vxGPU,
										vyGPU,
										vzGPU,
										cGPU);
    
    step++;
  }

  freeRandomNumbersGPU();
  //Free FFT
  cufftDestroy(FFT);

  if(!gpuToHostIncompressible()) return 0;
  return 1;
}


