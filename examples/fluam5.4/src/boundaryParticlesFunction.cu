// Filename: boundaryParticlesFunction.cu
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


void boundaryParticlesFunction(int numBlocksBoundary, 
			       int threadsPerBlockBoundary,
			       int numBlocksNeighbors, 
			       int threadsPerBlockNeighbors,
			       int numBlocksPartAndBoundary, 
			       int threadsPerBlockPartAndBoundary,
			       int numBlocksParticles, 
			       int threadsPerBlockParticles,
			       int numBlocks, 
			       int threadsPerBlock){
                                                         
  countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
  if(setboundary){
    forceBoundary<<<numBlocksBoundary,threadsPerBlockBoundary>>>(rxboundaryGPU,ryboundaryGPU,rzboundaryGPU,
								 vxboundaryGPU,vyboundaryGPU,vzboundaryGPU,
								 volumeboundaryGPU,
								 vxGPU,vyGPU,vzGPU,
								 rxcellGPU,rycellGPU,rzcellGPU,
								 fxboundaryGPU,fyboundaryGPU,fzboundaryGPU,
								 pc,errorKernel);
  }

  if(setparticles){
    findNeighborParticles<<<numBlocksPartAndBoundary,threadsPerBlockPartAndBoundary>>>(pc,errorKernel,
										       rxboundaryGPU,
										       ryboundaryGPU,
										       rzboundaryGPU,
										       vxboundaryGPU,
										       vyboundaryGPU,
										       vzboundaryGPU);
    
    /*nonBondedForce<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
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

    nonBondedForceExtraPressure<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
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
  }


  updateFluid<<<numBlocks,threadsPerBlock>>>(vxGPU,
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


  return;

}
