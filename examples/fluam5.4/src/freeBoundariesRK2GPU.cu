// Filename: freeBoundariesRK2GPU.cu
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


bool freeBoundariesRK2GPU(){

  cutilSafeCall(cudaUnbindTexture(texrxboundaryGPU));    
  cutilSafeCall(cudaUnbindTexture(texryboundaryGPU));    
  cutilSafeCall(cudaUnbindTexture(texrzboundaryGPU));
  //cutilSafeCall(cudaUnbindTexture(texfxboundaryGPU));
  //cutilSafeCall(cudaUnbindTexture(texfyboundaryGPU));
  //cutilSafeCall(cudaUnbindTexture(texfzboundaryGPU));

  cutilSafeCall(cudaFree(rxboundaryGPU));
  cutilSafeCall(cudaFree(ryboundaryGPU));
  cutilSafeCall(cudaFree(rzboundaryGPU));
  cutilSafeCall(cudaFree(rxboundaryPredictionGPU));
  cutilSafeCall(cudaFree(ryboundaryPredictionGPU));
  cutilSafeCall(cudaFree(rzboundaryPredictionGPU));
  cutilSafeCall(cudaFree(vxboundaryGPU));
  cutilSafeCall(cudaFree(vyboundaryGPU));
  cutilSafeCall(cudaFree(vzboundaryGPU));
  cutilSafeCall(cudaFree(vxboundaryPredictionGPU));
  cutilSafeCall(cudaFree(vyboundaryPredictionGPU));
  cutilSafeCall(cudaFree(vzboundaryPredictionGPU));
  cutilSafeCall(cudaFree(fxboundaryGPU));
  cutilSafeCall(cudaFree(fyboundaryGPU));
  cutilSafeCall(cudaFree(fzboundaryGPU));

  cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellX));
  cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellY));
  cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellZ));
  cutilSafeCall(cudaUnbindTexture(texPartInCellX));
  cutilSafeCall(cudaUnbindTexture(texPartInCellY));
  cutilSafeCall(cudaUnbindTexture(texPartInCellZ));
  cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellNonBonded));
  cutilSafeCall(cudaUnbindTexture(texPartInCellNonBonded));



  if(setparticles){
    cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellNonBonded));
    cutilSafeCall(cudaFree(countPartInCellNonBonded));

    cutilSafeCall(cudaUnbindTexture(texPartInCellNonBonded));
    cutilSafeCall(cudaFree(partInCellNonBonded));

    cutilSafeCall(cudaUnbindTexture(texneighbor0GPU));
    cutilSafeCall(cudaUnbindTexture(texneighbor1GPU));
    cutilSafeCall(cudaUnbindTexture(texneighbor2GPU));
    cutilSafeCall(cudaUnbindTexture(texneighbor3GPU));
    cutilSafeCall(cudaUnbindTexture(texneighbor4GPU));
    cutilSafeCall(cudaUnbindTexture(texneighbor5GPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxpyGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxmyGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxpzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxmzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxpyGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxmyGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxpzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxmzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpypzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpymzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormypzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormymzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxpypzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxpymzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxmypzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighborpxmymzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxpypzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxpymzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxmypzGPU));
    cutilSafeCall(cudaUnbindTexture(texneighbormxmymzGPU));
    cutilSafeCall(cudaFree(neighbor0GPU));
    cutilSafeCall(cudaFree(neighbor1GPU));
    cutilSafeCall(cudaFree(neighbor2GPU));
    cutilSafeCall(cudaFree(neighbor3GPU));
    cutilSafeCall(cudaFree(neighbor4GPU));
    cutilSafeCall(cudaFree(neighbor5GPU));
    cutilSafeCall(cudaFree(neighborpxpyGPU));
    cutilSafeCall(cudaFree(neighborpxmyGPU));
    cutilSafeCall(cudaFree(neighborpxpzGPU));
    cutilSafeCall(cudaFree(neighborpxmzGPU));
    cutilSafeCall(cudaFree(neighbormxpyGPU));
    cutilSafeCall(cudaFree(neighbormxmyGPU));
    cutilSafeCall(cudaFree(neighbormxpzGPU));
    cutilSafeCall(cudaFree(neighbormxmzGPU));
    cutilSafeCall(cudaFree(neighborpypzGPU));
    cutilSafeCall(cudaFree(neighborpymzGPU));
    cutilSafeCall(cudaFree(neighbormypzGPU));
    cutilSafeCall(cudaFree(neighbormymzGPU));
    cutilSafeCall(cudaFree(neighborpxpypzGPU));
    cutilSafeCall(cudaFree(neighborpxpymzGPU));
    cutilSafeCall(cudaFree(neighborpxmypzGPU));
    cutilSafeCall(cudaFree(neighborpxmymzGPU));
    cutilSafeCall(cudaFree(neighbormxpypzGPU));
    cutilSafeCall(cudaFree(neighbormxpymzGPU));
    cutilSafeCall(cudaFree(neighbormxmypzGPU));
    cutilSafeCall(cudaFree(neighbormxmymzGPU));
  }

  freeErrorArray();
  cutilSafeCall(cudaFree(pc));
  freeDelta();

  if(setparticles){
    cutilSafeCall(cudaUnbindTexture(texforceNonBonded1));
    cutilSafeCall(cudaFreeArray(forceNonBonded1));
  }

  //No-slip Test
  //cutilSafeCall(cudaFree(saveForceX));
  //cutilSafeCall(cudaFree(saveForceY));
  //cutilSafeCall(cudaFree(saveForceZ));

  cout << "FREE BOUNDARIES GPU :           DONE" << endl; 

  return 1;
}
