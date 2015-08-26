// Filename: freeMemoryRK3GPU.cu
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


bool freeMemoryRK3GPU(){
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

  cudaFree(dmGPU);
  cudaFree(dpxGPU);
  cudaFree(dpyGPU);
  cudaFree(dpzGPU);

  cudaFree(rxcellGPU);
  cudaFree(rycellGPU);
  cudaFree(rzcellGPU);

  
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
  cudaFree(stepGPU);

  cout << "FREE MEMORY GPU :               DONE" << endl; 


  return 1;
}

