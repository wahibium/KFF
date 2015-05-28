// Filename: boundaryParticlesFunctionparticlesWall.cu
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


void boundaryParticlesFunctionParticlesWall(int option,
					    int numBlocksBoundary, 
					    int threadsPerBlockBoundary,
					    int numBlocksNeighbors, 
					    int threadsPerBlockNeighbors,
					    int numBlocksPartAndBoundary, 
					    int threadsPerBlockPartAndBoundary,
					    int numBlocksParticles, 
					    int threadsPerBlockParticles,
					    int numBlocks, 
					    int threadsPerBlock,
					    int numBlocksCellst,
					    int threadsPerBlockCellst){
                                                         
  


  if(option==0){
    //Option=0
    //update particles to q^{n+1/2}
    //and spread particles forces
    
    countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
    
    if(setboundary){
      forceBoundary<<<numBlocksBoundary,threadsPerBlockBoundary>>>(rxboundaryGPU,
								   ryboundaryGPU,
								   rzboundaryGPU,
								   vxboundaryGPU,
								   vyboundaryGPU,
								   vzboundaryGPU,
								   volumeboundaryGPU,
								   vxGPU,
								   vyGPU,
								   vzGPU,
								   rxcellGPU,
								   rycellGPU,
								   rzcellGPU,
								   fxboundaryGPU,
								   fyboundaryGPU,
								   fzboundaryGPU,
								   pc,
								   errorKernel);
    }

    if(setparticles){
      //Update particle positions q^{n+1/2} = q^n + 0.5*dt * J^n * v^n
      //And fill partInCellNonBonded
      findNeighborParticlesQuasiNeutrallyBuoyant_1<<<numBlocksParticles,threadsPerBlockParticles>>>
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
      

      //Fill "countparticlesincellX" lists
      //and spread particle force F 
      kernelSpreadParticlesForce<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										  rycellGPU,
										  rzcellGPU,
										  fxboundaryGPU,
										  fyboundaryGPU,
										  fzboundaryGPU,
										  pc,errorKernel,
										  bFV);
    }

  }
  else{









    //Option=1
    //apply no-slip
    //update fluid and
    //update particles
    if(setparticles){
      countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
      
      nonBondedForceCompressibleParticles<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
											   rycellGPU,
											   rzcellGPU,
											   vxboundaryGPU,
											   vyboundaryGPU,
											   vzboundaryGPU,
											   densityGPU,
											   fxboundaryGPU,
											   fyboundaryGPU,
											   fzboundaryGPU,
											   pc,
											   errorKernel,
											   saveForceX,
											   saveForceY,
											   saveForceZ);

      /*nonBondedForceCompressibleParticlesExtraPressure<<<numBlocksParticles,threadsPerBlockParticles>>>
	(rxcellGPU,
	 rycellGPU,
	 rzcellGPU,
	 vxboundaryGPU,
	 vyboundaryGPU,
	 vzboundaryGPU,
	 densityGPU,
	 fxboundaryGPU,
	 fyboundaryGPU,
	 fzboundaryGPU,
	 pc,
	 errorKernel,
	 saveForceX,
	 saveForceY,
	 saveForceZ);*/

    }

    updateFluid<<<numBlocksCellst,threadsPerBlockCellst>>>(vxGPU,
							   vyGPU,
							   vzGPU,
							   fxboundaryGPU,
							   fyboundaryGPU,
							   fzboundaryGPU,
							   rxcellGPU,
							   rycellGPU,
							   rzcellGPU,
							   rxboundaryGPU,
							   ryboundaryGPU,
							   rzboundaryGPU,
							   pc);

    if(setparticles){
      //Calculate v^{n+0.5}
      //Store it in vxPredictionGPU
      calculateVelocityAtHalfTimeStepCompressibleParticles<<<numBlocksCellst,threadsPerBlockCellst>>>
	(vxGPU,//v^{n+1}
	 vyGPU,
	 vzGPU,
	 vxPredictionGPU,//v^n and v^{n+1/2}
	 vyPredictionGPU,
	 vzPredictionGPU);
      
      //Update particle position q^{n+1} = q^n + dt * J^{n+1/2} v^{n+1/2}
      findNeighborParticlesQuasiNeutrallyBuoyantTEST4_2<<<numBlocksParticles,threadsPerBlockParticles>>>
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
	 vxPredictionGPU, // v^{n+1/2}
	 vyPredictionGPU, 
	 vzPredictionGPU);
      
      //Load textures with particles position q^{n+1/2}
      cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
      cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
      cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));
      
    }
  }
  
  return;

}
