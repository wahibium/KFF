// Filename: runSchemeQuasiNeutrallyBuoyant.cu
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


bool runSchemeQuasiNeutrallyBuoyant2D(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

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
  size_t numberRandom = 6 * ncells;
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
					     expKz,pF);
  
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);













  // A. Donev: Project the initial velocity to make sure it is div-free
  //---------------------------------------------------------
  //Copy velocities to complex variable
  doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,vxZ,vyZ,vzZ);

  //Take velocities to fourier space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);

  //Project into divergence free space
  projectionDivergenceFree2D<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);

  //Take velocities to real space
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

  //Copy velocities to real variables
  doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);
  //---------------------------------------------------------


















  //First step. We use mid-point rule for the advection in the
  //first step, after that we continue with the 
  //Adams-Bashforth rule
  if(!firstStepQuasiNeutrallyBuoyant2D(numBlocksNeighbors,
				       threadsPerBlockNeighbors,
				       numBlocksParticles,
				       threadsPerBlockParticles,
				       numBlocks,
				       threadsPerBlock,
				       numberRandom,
				       FFT,
				       step)) return 0;

  while(step<numsteps){
    
    //
    //
    //
    //
    //
    //
    //Generate random numbers
    generateRandomNumbers(numberRandom);
    




    
    //
    //
    //
    //
    //
    //
    //STEP 1: UPDATE PARTICLE POSITIONS TO  q^{n+1/2}
    //Clear neighbor lists
    countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);

    //Update particle positions q^{n+1/2} = q^n + dt * J^n * v^n
    //saved in rxboundaryPredictionGPU
    findNeighborParticlesQuasiNeutrallyBuoyant_1_2D<<<numBlocksParticles,threadsPerBlockParticles>>>
      (pc, 
       errorKernel,
       rxcellGPU,
       rycellGPU,
       rzcellGPU,
       rxboundaryGPU,  //q^{n}
       ryboundaryGPU, 
       rzboundaryGPU,
       rxboundaryPredictionGPU, //q^{n+1/2}
       ryboundaryPredictionGPU, 
       rzboundaryPredictionGPU,
       vxGPU, //v^n
       vyGPU, 
       vzGPU);

    //Load textures with particles position q^{n+1/2}
    cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));






    //
    //
    //
    //
    //
    //
    //STEP 2: CALCULATE FORCES AND SPREAD THEM TO THE FLUID S^{n+1/2} * F^{n+1/2}
    //Fill "countparticlesincellX" lists
    //and spread particle force F 
    kernelSpreadParticlesForce2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										  rycellGPU,
										  rzcellGPU,
										  fxboundaryGPU,
										  fyboundaryGPU,
										  fzboundaryGPU,
										  pc,
										  errorKernel,
										  bFV);    






    //
    //
    //
    //
    //
    //
    //STEP 3: SOLVE UNPERTURBED FLUID MOMENTUM
    //Construct vector W
    // W = v^n + 0.5*dt*nu*L*v^n + Advection(v^n) + (dt/rho)*f^n_{noise} + dt*SF/rho + dt*(m_e/rho)*div*Suu
    //and save advection
    kernelConstructWQuasiNeutrallyBuoyantTEST5_3_2D<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
										   vyPredictionGPU,
										   vzPredictionGPU,
										   vxZ,//W
										   vyZ,//W
										   vzZ,//W
										   dRand,
										   fxboundaryGPU,
										   fyboundaryGPU,
										   fzboundaryGPU,
										   advXGPU,
										   advYGPU,
										   advZGPU);
									      
    
    //Calculate velocity prediction with incompressibility "\tilde{v}^{n+1}"
    //Go to fourier space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    //Apply shift for the staggered grid
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    //Update fluid velocity
    kernelUpdateVIncompressible2D<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W
    //Apply shift for the staggered grid
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    //Come back to real space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

    //Store velocity prediction "\tilde{v}^{n+1}" on vxPredictionGPU
    predictionVQuasiNeutrallyBuoyant<<<numBlocks,threadsPerBlock>>>(vxZ,
								    vyZ,
								    vzZ,
								    vxGPU,
								    vyGPU,
								    vzGPU,
								    vxPredictionGPU,
								    vyPredictionGPU,
								    vzPredictionGPU);
    
    //Load textures with velocity prediction "\tilde{v}^{n+1}"
    cutilSafeCall( cudaBindTexture(0,texVxGPU,vxPredictionGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVyGPU,vyPredictionGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVzGPU,vzPredictionGPU,ncells*sizeof(double)));






    //
    //
    //
    //
    //
    //
    //STEP 4: IF EXCESS OF MASS != 0 CALCULATE 
    //\delta u^{n+1/2} = (J^{n+1/2} - J^{n}) * v^n + 0.5*nu*dt * J^{n-1/2} * L * \Delta v^{n-1/2}
    if(mass != 0){
      //Calculate \delta u^{n+1/2} and saved in vxboundaryPredictionGPU
      kernelCalculateDeltau_2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										rycellGPU,
										rzcellGPU,
										vxGPU,
										vyGPU,
										vzGPU,
										rxboundaryGPU,
										ryboundaryGPU,
										rzboundaryGPU,
										vxboundaryPredictionGPU,
										vyboundaryPredictionGPU,
										vzboundaryPredictionGPU);
    }




    

    //
    //
    //
    //
    //
    //
    //STEP 5: IF EXCESS OF MASS != 0 CALCULATE
    //\Delta p
    if(mass != 0 ){
      //Calculate \Delta p and saved in vxboundaryGPU
      kernelCalculateDeltap_2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										rycellGPU,
										rzcellGPU,
										vxboundaryGPU,
										vyboundaryGPU,
										vzboundaryGPU,
										vxboundaryPredictionGPU,
										vyboundaryPredictionGPU,
										vzboundaryPredictionGPU);
    }






    //
    //
    //
    //
    //
    //
    //STEP 6: IF EXCESS OF MASS != 0 CALCULATE VELOCITY CORRECTION
    //\Delta \tilde{ v }
    if(mass != 0){
      //First, spread \Delta p, prefactor * S*{\Delta p}
      kernelSpreadDeltap2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
									    rycellGPU,
									    rzcellGPU,
									    vxboundaryGPU,
									    vyboundaryGPU,
									    vzboundaryGPU,
									    fxboundaryGPU,
									    fyboundaryGPU,
									    fzboundaryGPU);

      //Second, add all terms prefactor*S*{\Delta p} and store in vxZ.x
      kernelCorrectionVQuasiNeutrallyBuoyant_2_2D<<<numBlocks,threadsPerBlock>>>(vxZ,
										 vyZ,
										 vzZ,
										 fxboundaryGPU,
										 fyboundaryGPU,
										 fzboundaryGPU);
      
      //Third apply incompressibility to calculate \Delta \tilde{ v }
      cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
      cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
      //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
      kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
      projectionDivergenceFree2D<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
      kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
      cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
      cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
      //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
      //Store \Delta \tilde{ v } in vxZ.y
      saveDeltaTildev<<<numBlocks,threadsPerBlock>>>(vxZ,
						     vyZ,
						     vzZ);
    }
    





    //
    //
    //
    //
    //
    //
    //STEP 7: IF EXCESS OF MASS != 0 CALCULATE VELOCITY CORRECTION
    //\Delta v
    if(mass != 0){
      //First, spread S*(\Delta p - m_e*J*\Delta \tilde{ v })
      kernelSpreadDeltapMinusJTildev_2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
											 rycellGPU,
											 rzcellGPU,
											 vxZ,
											 vyZ,
											 vzZ,
											 vxboundaryGPU,
											 vyboundaryGPU,
											 vzboundaryGPU,
											 fxboundaryGPU,
											 fyboundaryGPU,
											 fzboundaryGPU);
      //The calculation of \Delta v has to wait, first we will
      //update the particle velocity
    }







    //
    //
    //
    //
    //
    //
    //STEP 8: UPDATE PARTICLE VELOCITY
    if(mass == 0){
      updateParticleVelocityme0_2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										    rycellGPU,
										    rzcellGPU,
										    vxboundaryGPU,
										    vyboundaryGPU,
										    vzboundaryGPU);
    }
    else{
      updateParticleVelocityme_2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										   rycellGPU,
										   rzcellGPU,
										   vxZ,
										   vyZ,
										   vzZ,
										   vxboundaryGPU,
										   vyboundaryGPU,
										   vzboundaryGPU,
										   vxboundaryPredictionGPU,
										   vyboundaryPredictionGPU,
										   vzboundaryPredictionGPU);
    }





    //
    //
    //
    //
    //
    //
    //STEP 9: IF EXCESS OF MASS != 0 CALCULATE VELOCITY CORRECTION
    //\Delta v
    if(mass != 0){
      //We finish here what we started in STEP 7.
      
      //Add all terms S*({\Delta p} - me*J*\tilde{v}) and store in vxZ.x
      kernelCorrectionVQuasiNeutrallyBuoyant_2_2D<<<numBlocks,threadsPerBlock>>>(vxZ,
										 vyZ,
										 vzZ,
										 fxboundaryGPU,
										 fyboundaryGPU,
										 fzboundaryGPU);

      //Third apply incompressibility to calculate \Delta \tilde{ v }
      cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
      cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
      //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
      kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
      //projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
      kernelUpdateVIncompressible2D<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W 
      kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
      cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
      cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
      //cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

      //The result \Delta v is store in vxZ.x without normalization
    }


    
    
    
    //
    //
    //
    //
    //
    //
    //STEP 10: UPDATE PARTICLE POSITION
    //q^{n+1} = q^n + 0.5*dt * J^{n+1/2} * (v^n + v^{n+1})
    //Calculate v^{n+0.5}
    //Store it in vxGPU
    if(mass != 0){
      calculateVelocityAtHalfTimeStep<<<numBlocks,threadsPerBlock>>>(vxGPU,
								     vyGPU,
								     vzGPU,
								     vxPredictionGPU,
								     vyPredictionGPU,
								     vzPredictionGPU,
								     vxZ,
								     vyZ,
								     vzZ);
    }
    else{
      calculateVelocityAtHalfTimeStepme0<<<numBlocks,threadsPerBlock>>>(vxGPU,
									vyGPU,
									vzGPU,
									vxPredictionGPU,
									vyPredictionGPU,
									vzPredictionGPU);
    }
    
    //Update particle position q^{n+1} = q^n + dt * J^{n+1/2} v^{n+1/2}
    findNeighborParticlesQuasiNeutrallyBuoyantTEST4_2_2D<<<numBlocksParticles,threadsPerBlockParticles>>>
      (pc, 
       errorKernel,
       rxcellGPU,
       rycellGPU,
       rzcellGPU,
       rxboundaryGPU, //q^{n} and q^{n+1}
       ryboundaryGPU,
       rzboundaryGPU,
       rxboundaryPredictionGPU, //q^{n+1/2}
       ryboundaryPredictionGPU, 
       rzboundaryPredictionGPU,
       vxGPU, // v^{n+1/2}
       vyGPU,
       vzGPU);





    //
    //
    //
    //
    //
    //
    //STEP 11: IF EXCESS OF MASS !=0 CALCULATE LAGGING TERM FOR THE NEXT STEP
    //0.5*nu*dt * J^{n-1/2} * \Delta v
    if(mass != 0){
      //Calculate 0.5*dt*nu*L*\Delta v and store it
      //in vxGPU
      laplacianDeltaV_2D<<<numBlocks,threadsPerBlock>>>(vxZ,
							vyZ,
							vzZ,
							vxGPU,
							vyGPU,
							vzGPU);
      
      //Calculate 0.5*dt*nu*J*L*\Delta v and store it
      //in vxboundaryPredictionGPU
      interpolateLaplacianDeltaV_2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										     rycellGPU,
										     rzcellGPU,
										     vxGPU,
										     vyGPU,
										     vzGPU,
										     rxboundaryPredictionGPU,
										     ryboundaryPredictionGPU,
										     rzboundaryPredictionGPU,
										     vxboundaryPredictionGPU,
										     vyboundaryPredictionGPU,
										     vzboundaryPredictionGPU);
    }



    //Load textures with particles position q^{n}
    cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));



    //
    //
    //
    //
    //
    //
    //STEP 12: UPDATE FLUID VELOCITY
    //Update fluid velocity
    if(mass == 0){
      updateFluidme0<<<numBlocks,threadsPerBlock>>>(vxGPU,
						    vyGPU,
						    vzGPU,
						    vxPredictionGPU,
						    vyPredictionGPU,
						    vzPredictionGPU);
    }
    else{
      updateFluidQuasiNeutrallyBuoyantSemiImplicit<<<numBlocks,threadsPerBlock>>>(vxGPU,
										  vyGPU,
										  vzGPU,
										  vxPredictionGPU,
										  vyPredictionGPU,
										  vzPredictionGPU,
										  vxZ,
										  vyZ,
										  vzZ);
    }

    //Load textures with velocity prediction "\tilde{v}^{n+1}"
    cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncells*sizeof(double)));








								    
    step++;
    if(!(step%samplefreq)&&(step>0)){
      cout << "QuasiNeutrallyBuoyant 2D " << step << endl;
      if(!gpuToHostIncompressibleBoundaryRK2(numBlocksParticles,threadsPerBlockParticles)) return 0;
      if(!saveFunctionsSchemeIncompressibleBoundary2D(1,step)) return 0;
    }
  }



  //Free FFT
  cufftDestroy(FFT);
  freeRandomNumbersGPU();

  

  return 1;
}
