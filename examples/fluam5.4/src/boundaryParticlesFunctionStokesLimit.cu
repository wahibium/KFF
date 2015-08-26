// Filename: boundaryParticlesFunctionStokesLimit.cu
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


void boundaryParticlesFunctionStokesLimit(int option,
					  int numBlocksBoundary, 
					  int threadsPerBlockBoundary,
					  int numBlocksNeighbors, 
					  int threadsPerBlockNeighbors,
					  int numBlocksPartAndBoundary, 
					  int threadsPerBlockPartAndBoundary,
					  int numBlocksParticles, 
					  int threadsPerBlockParticles,
					  int numBlocks, 
					  int threadsPerBlock){

  


  if(option==0){
    //Option=0
    //update particles to q^{n+1/2}
    //and spread particles forces

    countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
    
    if(setparticles){

      //Fill partInCellNonBonded
      findNeighborParticlesStokesLimit_1<<<numBlocksParticles,threadsPerBlockParticles>>>
	(pc, 
	 errorKernel);
	 
     
      //Fill "countparticlesincellX" lists
      //and spread particle force F 
      kernelSpreadParticlesForceStokesLimit<<<numBlocksParticles,threadsPerBlockParticles>>>
	(rxcellGPU,
	 rycellGPU,
	 rzcellGPU,
	 fxboundaryGPU,
	 fyboundaryGPU,
	 fzboundaryGPU,
	 vxboundaryGPU,//Fx, this is the force on the particle.
	 vyboundaryGPU,
	 vzboundaryGPU,
	 pc,errorKernel,
	 bFV);

      //Set vxGPU to zero
      setFieldToZeroInput<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU);

      //Add force spreaded by the particles
      addSpreadedForcesStokesLimit<<<numBlocks,threadsPerBlock>>>(vxGPU, //Stored fx=S*Fx
								  vyGPU,
								  vzGPU,
								  fxboundaryGPU,
								  fyboundaryGPU,
								  fzboundaryGPU,
								  pc);
      
      countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);

      //Spread particle drift term to positive directions
      //(kT/delta) * [S(q+0.5*delta*W)*W]
      kernelSpreadParticlesDrift<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										  rycellGPU,
										  rzcellGPU,
										  fxboundaryGPU,
										  fyboundaryGPU,
										  fzboundaryGPU,
										  dRand,
										  pc,
										  1);

      //Add drift spreaded by the particles
      addSpreadedForcesStokesLimit<<<numBlocks,threadsPerBlock>>>(vxGPU, //Stored fx=S*Fx+S*drift_p
								  vyGPU,
								  vzGPU,
								  fxboundaryGPU,
								  fyboundaryGPU,
								  fzboundaryGPU,
								  pc);

      
      countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);

      //Spread particle drift term to negative directions
      //(kT/delta) * [-S(q-0.5*delta*W)*W]
      kernelSpreadParticlesDrift<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										  rycellGPU,
										  rzcellGPU,
										  fxboundaryGPU,
										  fyboundaryGPU,
										  fzboundaryGPU,
										  dRand,
										  pc,
										  -1);

      //Add drift spreaded by the particles
      addSpreadedForcesStokesLimit<<<numBlocks,threadsPerBlock>>>(vxGPU, //Stored fx=S*Fx+S*drift_p-S*drift_m
								  vyGPU,
								  vzGPU,
								  fxboundaryGPU,
								  fyboundaryGPU,
								  fzboundaryGPU,
								  pc);

    }

  }
  else{




    //Option=1
    //apply no-slip
    //update fluid and
    //update particles
    if(setparticles){

      countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
      
      //Update particle position half time step
      // q^{n+1/2} = q^n + 0.5*dt*extraMobility*F^n + 0.5*dt*J^n * v + noise_1
      //And fill partInCellNonBonded
      updateParticlesStokesLimit_1<<<numBlocksParticles,threadsPerBlockParticles>>>
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
	 vxboundaryGPU,//Fx^n
	 vyboundaryGPU,
	 vzboundaryGPU,
	 vxGPU, //
	 vyGPU, 
	 vzGPU,
	 dRand);

      //Load textures with particles position q^{n+1/2}
      cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
      cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
      cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));

      //Calculate particles force at (n+1/2)*dt
      if(setExtraMobility){
	particlesForceStokesLimit<<<numBlocksParticles,threadsPerBlockParticles>>>
	  (vxboundaryGPU,//Fx, this is the force on the particle.
	   vyboundaryGPU,
	   vzboundaryGPU,
	   pc,
	   errorKernel,
	   bFV);
      }


      //Update particle position
      // q^{n+1} = q^n + dt*extraMobility*F^{n+1/2} + dt*J^{n+1/2} * v + noise_1 + noise_2   
      updateParticlesStokesLimit_2<<<numBlocksParticles,threadsPerBlockParticles>>>
	(rxcellGPU,
	 rycellGPU,
	 rzcellGPU,
	 rxboundaryGPU,  //q^{n}
	 ryboundaryGPU, 
	 rzboundaryGPU,
	 vxboundaryGPU, //Fx^{n+1/2}
	 vyboundaryGPU,
	 vzboundaryGPU,
	 vxGPU, //
	 vyGPU, 
	 vzGPU,
	 dRand,
	 pc, 
	 errorKernel);
      
      //Load textures with particles position q^{n+1}
      cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
      cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
      cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));


    }

  }
  
  return;

}
