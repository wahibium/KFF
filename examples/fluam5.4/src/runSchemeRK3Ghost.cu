// Filename: runSchemeRK3Ghost.cu
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


bool runSchemeRK3Ghost(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  int nGhost = ncellst - ncells;
  int threadsPerBlockGhost = 128;
  if((nGhost/threadsPerBlockGhost) < 60) threadsPerBlockGhost = 64;
  if((nGhost/threadsPerBlockGhost) < 60) threadsPerBlockGhost = 32;
  int numBlocksGhost = (nGhost-1)/threadsPerBlockGhost + 1;



  //Initialize textures cells
  if(!texturesCellsGhost()) return 0;
  
  //Inilialize ghost index
  if(!initGhostIndexGPU()) return 0;
  
  step = -numstepsRelaxation;


  while(step<numsteps){
    //Provide data to ghost cells
    kernelFeedGhostCellsRK3<<<numBlocksGhost,threadsPerBlockGhost>>>(ghostToPIGPU,
								     ghostToGhostGPU,
								     densityGPU,
								     densityPredictionGPU,
								     vxGPU,
								     vyGPU,
								     vzGPU,
								     vxPredictionGPU,
								     vyPredictionGPU,
								     vzPredictionGPU);
    //First substep RK3
    kernelDpRK3Ghost_1<<<numBlocks,threadsPerBlock>>>(densityGPU,
						      densityGPU,
						      vxGPU,
						      vyGPU,
						      vzGPU,
						      dmGPU,
						      dpxGPU,
						      dpyGPU,
						      dpzGPU,
						      ghostIndexGPU,
						      realIndexGPU,
						      0,1);

    kernelDpRK3Ghost_2<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,
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
    kernelFeedGhostCellsRK3<<<numBlocksGhost,threadsPerBlockGhost>>>(ghostToPIGPU,
								     ghostToGhostGPU,
								     densityGPU,
								     densityPredictionGPU,
								     vxGPU,
								     vyGPU,
								     vzGPU,
								     vxPredictionGPU,
								     vyPredictionGPU,
								     vzPredictionGPU);

    //Second substep RK3
    kernelDpRK3Ghost_1<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,
						      densityGPU,
						      vxGPU,
						      vyGPU,
						      vzGPU,
						      dmGPU,
						      dpxGPU,
						      dpyGPU,
						      dpzGPU,
						      ghostIndexGPU,
						      realIndexGPU,
						      0.75,0.25);

    kernelDpRK3Ghost_2<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,
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
    kernelFeedGhostCellsRK3<<<numBlocksGhost,threadsPerBlockGhost>>>(ghostToPIGPU,
								     ghostToGhostGPU,
								     densityGPU,
								     densityPredictionGPU,
								     vxGPU,
								     vyGPU,
								     vzGPU,
								     vxPredictionGPU,
								     vyPredictionGPU,
								     vzPredictionGPU);

    //Third substep RK3
    kernelDpRK3Ghost_1<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,
						      densityGPU,
						      vxGPU,
						      vyGPU,
						      vzGPU,
						      dmGPU,
						      dpxGPU,
						      dpyGPU,
						      dpzGPU,
						      ghostIndexGPU,
						      realIndexGPU,
						      1./3.,2./3.);

    kernelDpRK3Ghost_2<<<numBlocks,threadsPerBlock>>>(densityGPU,
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
    
    
    step++;
    //cout << step << endl;

    if(!(step%samplefreq)&&(step>0)){
      kernelFeedGhostCellsRK3<<<numBlocksGhost,threadsPerBlockGhost>>>(ghostToPIGPU,
								       ghostToGhostGPU,
								       densityGPU,
								       densityPredictionGPU,
								       vxGPU,
								       vyGPU,
								       vzGPU,
								       vxPredictionGPU,
								       vyPredictionGPU,
								       vzPredictionGPU);

      cout << "RK3 " << step << endl;
      if(!gpuToHostRK3Ghost()) return 0;
      if(!saveFunctionsSchemeRK3Ghost(1)) return 0;
    }
  }
  
    






  return 1;
}
