// Filename: runSchemeParticlesWall.cu
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


bool runSchemeParticlesWall(){
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
  if(ncellst>numNeighbors){
    threadsPerBlockNeighbors = 128;
    if((ncellst/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 64;
    if((ncellst/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 32;
    numBlocksNeighbors = (ncellst-1)/threadsPerBlockNeighbors + 1;
  }
  else{
    threadsPerBlockNeighbors = 128;
    if((numNeighbors/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 64;
    if((numNeighbors/threadsPerBlockNeighbors) < 60) threadsPerBlockNeighbors = 32;
    numBlocksNeighbors = (numNeighbors-1)/threadsPerBlockNeighbors + 1;
  }
  int nGhost = ncellst - ncells;
  int threadsPerBlockGhost = 128;
  if((nGhost/threadsPerBlockGhost) < 60) threadsPerBlockGhost = 64;
  if((nGhost/threadsPerBlockGhost) < 60) threadsPerBlockGhost = 32;
  int numBlocksGhost = (nGhost-1)/threadsPerBlockGhost + 1;

  int threadsPerBlockCellst = 128;
  if((ncellst/threadsPerBlockCellst) < 60) threadsPerBlockCellst = 64;
  if((ncellst/threadsPerBlockCellst) < 60) threadsPerBlockCellst = 32;
  int numBlocksCellst = (ncellst-1)/threadsPerBlockCellst + 1;











  //initialize random numbers
  size_t numberRandom = 12 * ncellst;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;


  //Initialize textures cells
  if(!texturesCells()) return 0;
  
  //Inilialize ghost index
  if(!initGhostIndexParticlesWallGPU()) return 0;


  initializeVecinos<<<numBlocksCellst,threadsPerBlockCellst>>>(vecino1GPU,
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
  
  initializeVecinos2<<<numBlocksCellst,threadsPerBlockCellst>>>(vecino0GPU,
								vecino1GPU,
								vecino2GPU,
								vecino3GPU,
								vecino4GPU,
								vecino5GPU);


  int substep = 0;
  step = -numstepsRelaxation;
  













  while(step<numsteps){
    //Generate random numbers
    generateRandomNumbers(numberRandom);

    //Provide data to ghost cells
    /*kernelFeedGhostCellsParticlesWall<<<numBlocksGhost,threadsPerBlockGhost>>>
      (ghostToPIGPU,
       ghostToGhostGPU,
       densityGPU,
       densityPredictionGPU,
       vxGPU,
       vyGPU,
       vzGPU,
       vxPredictionGPU,
       vyPredictionGPU,
       vzPredictionGPU,
       dRand);*/

    kernelFeedGhostCellsParticlesWall2<<<numBlocksGhost,threadsPerBlockGhost>>>
	(ghostToPIGPU,
	 ghostToGhostGPU,
	 densityGPU,
	 densityPredictionGPU,
	 vxGPU,
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,
	 vyPredictionGPU,
	 vzPredictionGPU,
	 dRand);

    //Boundaries and particles 
    //Update particles position to q^{n+1/2}
    //Spread and interpolate particles force
    boundaryParticlesFunctionParticlesWall(0,
					   numBlocksBoundary,
					   threadsPerBlockBoundary,
					   numBlocksNeighbors,
					   threadsPerBlockNeighbors,
					   numBlocksPartAndBoundary,
					   threadsPerBlockPartAndBoundary,
					   numBlocksParticles,
					   threadsPerBlockParticles,
					   numBlocks,
					   threadsPerBlock,
					   numBlocksCellst,
					   threadsPerBlockCellst);

    //Provide data to ghost cells
    /*kernelFeedGhostCellsParticlesWall<<<numBlocksGhost,threadsPerBlockGhost>>>
      (ghostToPIGPU,
       ghostToGhostGPU,
       densityGPU,
       densityPredictionGPU,
       vxGPU,
       vyGPU,
       vzGPU,
       vxPredictionGPU,
       vyPredictionGPU,
       vzPredictionGPU,
       dRand);*/

    kernelFeedGhostCellsParticlesWall2<<<numBlocksGhost,threadsPerBlockGhost>>>
	(ghostToPIGPU,
	 ghostToGhostGPU,
	 densityGPU,
	 densityPredictionGPU,
	 vxGPU,
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,
	 vyPredictionGPU,
	 vzPredictionGPU,
	 dRand);

    //First substep RK3
    kernelDpParticlesWall<<<numBlocksCellst,threadsPerBlockCellst>>>(densityGPU,
								     densityGPU,
								     vxGPU,
								     vyGPU,
								     vzGPU,
								     dmGPU,
								     dpxGPU,
								     dpyGPU,
								     dpzGPU,
								     dRand,
								     fxboundaryGPU,
								     fyboundaryGPU,
								     fzboundaryGPU,
								     ghostIndexGPU, 
								     realIndexGPU,
								     substep,
								     0,1,-sqrt(3));

    kernelDpParticlesWall_2<<<numBlocksCellst,threadsPerBlockCellst>>>(densityPredictionGPU,
								       vxPredictionGPU,
								       vyPredictionGPU,
								       vzPredictionGPU,
								       dmGPU,
								       dpxGPU,
								       dpyGPU,
								       dpzGPU,
								       ghostIndexGPU, 
								       realIndexGPU);
    
    cutilSafeCall( cudaBindTexture(0,texVxGPU,vxPredictionGPU,ncellst*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVyGPU,vyPredictionGPU,ncellst*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVzGPU,vzPredictionGPU,ncellst*sizeof(double)));

    //Provide data to ghost cells
    /*kernelFeedGhostCellsParticlesWall<<<numBlocksGhost,threadsPerBlockGhost>>>
      (ghostToPIGPU,
       ghostToGhostGPU,
       densityGPU,
       densityPredictionGPU,
       vxGPU,
       vyGPU,
       vzGPU,
       vxPredictionGPU,
       vyPredictionGPU,
       vzPredictionGPU,
       dRand);*/

    kernelFeedGhostCellsParticlesWall2<<<numBlocksGhost,threadsPerBlockGhost>>>
	(ghostToPIGPU,
	 ghostToGhostGPU,
	 densityGPU,
	 densityPredictionGPU,
	 vxGPU,
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,
	 vyPredictionGPU,
	 vzPredictionGPU,
	 dRand);
 
    //Second substep RK3
    kernelDpParticlesWall<<<numBlocksCellst,threadsPerBlockCellst>>>(densityPredictionGPU,
								     densityGPU,
								     vxGPU,
								     vyGPU,
								     vzGPU,
								     dmGPU,
								     dpxGPU,
								     dpyGPU,
								     dpzGPU,
								     dRand,
								     fxboundaryGPU,
								     fyboundaryGPU,
								     fzboundaryGPU,
								     ghostIndexGPU, 
								     realIndexGPU,
								     substep,
								     0.75,0.25,sqrt(3));
    
    kernelDpParticlesWall_2<<<numBlocksCellst,threadsPerBlockCellst>>>(densityPredictionGPU,
								       vxPredictionGPU,
								       vyPredictionGPU,
								       vzPredictionGPU,
								       dmGPU,
								       dpxGPU,
								       dpyGPU,
								       dpzGPU,
								       ghostIndexGPU, 
								       realIndexGPU);
    
    //Provide data to ghost cells
    /*kernelFeedGhostCellsParticlesWall<<<numBlocksGhost,threadsPerBlockGhost>>>
      (ghostToPIGPU,
       ghostToGhostGPU,
       densityGPU,
       densityPredictionGPU,
       vxGPU,
       vyGPU,
       vzGPU,
       vxPredictionGPU,
       vyPredictionGPU,
       vzPredictionGPU,
       dRand);*/

    kernelFeedGhostCellsParticlesWall2<<<numBlocksGhost,threadsPerBlockGhost>>>
	(ghostToPIGPU,
	 ghostToGhostGPU,
	 densityGPU,
	 densityPredictionGPU,
	 vxGPU,
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,
	 vyPredictionGPU,
	 vzPredictionGPU,
	 dRand);

    //Third substep RK3
    kernelDpParticlesWall<<<numBlocksCellst,threadsPerBlockCellst>>>(densityPredictionGPU,
								     densityGPU,
								     vxGPU,
								     vyGPU,
								     vzGPU,
								     dmGPU,
								     dpxGPU,
								     dpyGPU,
								     dpzGPU,
								     dRand,
								     fxboundaryGPU,
								     fyboundaryGPU,
								     fzboundaryGPU,
								     ghostIndexGPU, 
								     realIndexGPU,
								     substep,
								     1./3.,2./3.,0);

    //Copy v^n to vxPredictionGPU
    //We need it for the particle update
    copyField<<<numBlocksCellst,threadsPerBlockCellst>>>(vxGPU,
							 vyGPU,
							 vzGPU,
							 vxPredictionGPU,
							 vyPredictionGPU,
							 vzPredictionGPU);
    
    kernelDpParticlesWall_2<<<numBlocksCellst,threadsPerBlockCellst>>>(densityGPU,
								       vxGPU,
								       vyGPU,
								       vzGPU,
								       dmGPU,
								       dpxGPU,
								       dpyGPU,
								       dpzGPU,
								       ghostIndexGPU, 
								       realIndexGPU);

    cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncellst*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncellst*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncellst*sizeof(double)));


    //Boundaries and particles part start
    boundaryParticlesFunctionParticlesWall(1,
					   numBlocksBoundary,
					   threadsPerBlockBoundary,
					   numBlocksNeighbors,
					   threadsPerBlockNeighbors,
					   numBlocksPartAndBoundary,
					   threadsPerBlockPartAndBoundary,
					   numBlocksParticles,
					   threadsPerBlockParticles,
					   numBlocks,
					   threadsPerBlock,
					   numBlocksCellst,
					   threadsPerBlockCellst);




    step++;
        
    if(!(step%samplefreq)&&(step>0)){
      cout << "Particles Wall  " << step << endl;

      //Provide data to ghost cells
      /*kernelFeedGhostCellsParticlesWall<<<numBlocksGhost,threadsPerBlockGhost>>>
	(ghostToPIGPU,
	 ghostToGhostGPU,
	 densityGPU,
	 densityPredictionGPU,
	 vxGPU,
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,
	 vyPredictionGPU,
	 vzPredictionGPU,
	 dRand);*/
      
      kernelFeedGhostCellsParticlesWall2<<<numBlocksGhost,threadsPerBlockGhost>>>
	(ghostToPIGPU,
	 ghostToGhostGPU,
	 densityGPU,
	 densityPredictionGPU,
	 vxGPU,
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,
	 vyPredictionGPU,
	 vzPredictionGPU,
	 dRand);

      if(!gpuToHostParticles()) return 0;
      if(!saveFunctionsSchemeParticlesWall(1,step)) return 0;
      
    }
    
  }


  freeRandomNumbersGPU();

  return 1;
}
