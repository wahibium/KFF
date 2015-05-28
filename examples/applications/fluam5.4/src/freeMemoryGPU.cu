// Filename: freeMemoryGPU.cu
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


bool freeMemoryGPU(){
  if(thermostat == 1){
    cudaFree(d_rand);
  }
  //cudaFree(massGPU);
  cudaFree(densityGPU);
  cutilSafeCall(cudaUnbindTexture(texVxGPU));
  cutilSafeCall(cudaUnbindTexture(texVyGPU));
  cutilSafeCall(cudaUnbindTexture(texVzGPU));    
  cudaFree(vxGPU);
  cudaFree(vyGPU);
  cudaFree(vzGPU);
  cudaFree(densityPredictionGPU);
  cudaFree(vxPredictionGPU);
  cudaFree(vyPredictionGPU);
  cudaFree(vzPredictionGPU);

  //cudaFree(fxGPU);
  //cudaFree(fyGPU);
  //cudaFree(fzGPU);

  cudaFree(dmGPU);
  cudaFree(dpxGPU);
  cudaFree(dpyGPU);
  cudaFree(dpzGPU);

  cudaFree(rxcellGPU);
  cudaFree(rycellGPU);
  cudaFree(rzcellGPU);

  if((setboundary==1) || (setparticles==1)){
    cutilSafeCall(cudaUnbindTexture(texrxboundaryGPU));    
    cutilSafeCall(cudaUnbindTexture(texryboundaryGPU));    
    cutilSafeCall(cudaUnbindTexture(texrzboundaryGPU));
    cutilSafeCall(cudaUnbindTexture(texfxboundaryGPU));
    cutilSafeCall(cudaUnbindTexture(texfyboundaryGPU));
    cutilSafeCall(cudaUnbindTexture(texfzboundaryGPU));
    cudaFree(rxboundaryGPU);
    cudaFree(ryboundaryGPU);
    cudaFree(rzboundaryGPU);
    cutilSafeCall(cudaFree(vxboundaryGPU));
    cutilSafeCall(cudaFree(vyboundaryGPU));
    cutilSafeCall(cudaFree(vzboundaryGPU));
    cutilSafeCall(cudaFree(fxboundaryGPU));
    cutilSafeCall(cudaFree(fyboundaryGPU));
    cutilSafeCall(cudaFree(fzboundaryGPU));
    
    //freeDelta();

    cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellX));
    cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellY));
    cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellZ));
    cutilSafeCall(cudaUnbindTexture(texPartInCellX));
    cutilSafeCall(cudaUnbindTexture(texPartInCellY));
    cutilSafeCall(cudaUnbindTexture(texPartInCellZ));
    cutilSafeCall(cudaUnbindTexture(texCountParticlesInCellNonBonded));
    cutilSafeCall(cudaUnbindTexture(texPartInCellNonBonded));
  }

  if(setparticles == 1){
    cutilSafeCall(cudaFree(countPartInCellNonBonded));
    cutilSafeCall(cudaFree(partInCellNonBonded));
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
    cutilSafeCall(cudaUnbindTexture(texforceNonBonded1));
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
    cutilSafeCall(cudaFreeArray(forceNonBonded1));
    cutilSafeCall(cudaFree(saveForceX));
    cutilSafeCall(cudaFree(saveForceY));
    cutilSafeCall(cudaFree(saveForceZ));
  }
  
  cutilSafeCall(cudaUnbindTexture(texvecino0GPU));
  cutilSafeCall(cudaUnbindTexture(texvecino1GPU));
  cutilSafeCall(cudaUnbindTexture(texvecino2GPU));
  cutilSafeCall(cudaUnbindTexture(texvecino3GPU));
  cutilSafeCall(cudaUnbindTexture(texvecino4GPU));
  cutilSafeCall(cudaUnbindTexture(texvecino5GPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxpyGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxmyGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxpzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxmzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxpyGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxmyGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxpzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxmzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopypzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopymzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomypzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomymzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxpypzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxpymzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxmypzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinopxmymzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxpypzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxpymzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxmypzGPU));
  cutilSafeCall(cudaUnbindTexture(texvecinomxmymzGPU));

  cudaFree(vecino0GPU);
  cudaFree(vecino1GPU);
  cudaFree(vecino2GPU);
  cudaFree(vecino3GPU);
  cudaFree(vecino4GPU);
  cudaFree(vecino5GPU);

  cudaFree(vecinopxpyGPU);
  cudaFree(vecinopxmyGPU);
  cudaFree(vecinopxpzGPU);
  cudaFree(vecinopxmzGPU);
  cudaFree(vecinomxpyGPU);
  cudaFree(vecinomxmyGPU);
  cudaFree(vecinomxpzGPU);
  cudaFree(vecinomxmzGPU);  
  cudaFree(vecinopypzGPU);
  cudaFree(vecinopymzGPU);
  cudaFree(vecinomypzGPU);
  cudaFree(vecinomymzGPU);
  cudaFree(vecinopxpypzGPU);
  cudaFree(vecinopxpymzGPU);
  cudaFree(vecinopxmypzGPU);
  cudaFree(vecinopxmymzGPU);
  cudaFree(vecinomxpypzGPU);
  cudaFree(vecinomxpymzGPU);
  cudaFree(vecinomxmypzGPU);
  cudaFree(vecinomxmymzGPU);

  cudaFree(countparticlesincellX);
  cudaFree(countparticlesincellY);
  cudaFree(countparticlesincellZ);
  cudaFree(partincellX);
  cudaFree(partincellY);
  cudaFree(partincellZ);
  cutilSafeCall(cudaFree(errorKernel));
  cutilSafeCall(cudaFree(stepGPU));

  if(setCheckVelocity==1){
    cutilSafeCall(cudaFree(rxCheckGPU));
    cutilSafeCall(cudaFree(ryCheckGPU));
    cutilSafeCall(cudaFree(rzCheckGPU));
    cutilSafeCall(cudaFree(vxCheckGPU));
    cutilSafeCall(cudaFree(vyCheckGPU));
    cutilSafeCall(cudaFree(vzCheckGPU));
  }


  if(!freeOtherFluidVariablesGPU()) return 0;

  cout << "FREE MEMORY GPU :               DONE" << endl; 


  return 1;
}

