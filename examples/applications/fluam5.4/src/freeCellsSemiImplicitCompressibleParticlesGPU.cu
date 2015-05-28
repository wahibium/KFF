// Filename: freeCellsGPU.cu
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


bool freeCellsSemiImplicitCompressibleParticlesGPU(){
  cutilSafeCall(cudaFree(densityGPU));
  cutilSafeCall(cudaUnbindTexture(texVxGPU));
  cutilSafeCall(cudaUnbindTexture(texVyGPU));
  cutilSafeCall(cudaUnbindTexture(texVzGPU));    
  cutilSafeCall(cudaFree(vxGPU));
  cutilSafeCall(cudaFree(vyGPU));
  cutilSafeCall(cudaFree(vzGPU));
  cutilSafeCall(cudaFree(densityPredictionGPU));
  cutilSafeCall(cudaFree(vxPredictionGPU));
  cutilSafeCall(cudaFree(vyPredictionGPU));
  cutilSafeCall(cudaFree(vzPredictionGPU));

  cutilSafeCall(cudaFree(dmGPU));
  cutilSafeCall(cudaFree(dpxGPU));
  cutilSafeCall(cudaFree(dpyGPU));
  cutilSafeCall(cudaFree(dpzGPU));

  cutilSafeCall(cudaFree(rxcellGPU));
  cutilSafeCall(cudaFree(rycellGPU));
  cutilSafeCall(cudaFree(rzcellGPU));

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

  cutilSafeCall(cudaFree(vecino0GPU));
  cutilSafeCall(cudaFree(vecino1GPU));
  cutilSafeCall(cudaFree(vecino2GPU));
  cutilSafeCall(cudaFree(vecino3GPU));
  cutilSafeCall(cudaFree(vecino4GPU));
  cutilSafeCall(cudaFree(vecino5GPU));
  cutilSafeCall(cudaFree(vecinopxpyGPU));
  cutilSafeCall(cudaFree(vecinopxmyGPU));
  cutilSafeCall(cudaFree(vecinopxpzGPU));
  cutilSafeCall(cudaFree(vecinopxmzGPU));
  cutilSafeCall(cudaFree(vecinomxpyGPU));
  cutilSafeCall(cudaFree(vecinomxmyGPU));
  cutilSafeCall(cudaFree(vecinomxpzGPU));
  cutilSafeCall(cudaFree(vecinomxmzGPU));
  cutilSafeCall(cudaFree(vecinopypzGPU));
  cutilSafeCall(cudaFree(vecinopymzGPU));
  cutilSafeCall(cudaFree(vecinomypzGPU));
  cutilSafeCall(cudaFree(vecinomymzGPU));
  cutilSafeCall(cudaFree(vecinopxpypzGPU));
  cutilSafeCall(cudaFree(vecinopxpymzGPU));
  cutilSafeCall(cudaFree(vecinopxmypzGPU));
  cutilSafeCall(cudaFree(vecinopxmymzGPU));
  cutilSafeCall(cudaFree(vecinomxpypzGPU));
  cutilSafeCall(cudaFree(vecinomxpymzGPU));
  cutilSafeCall(cudaFree(vecinomxmypzGPU));
  cutilSafeCall(cudaFree(vecinomxmymzGPU));

  cutilSafeCall(cudaFree(stepGPU));


  cutilSafeCall(cudaFree(pF));

  cutilSafeCall(cudaFree(gradKx));
  cutilSafeCall(cudaFree(gradKy));
  cutilSafeCall(cudaFree(gradKz));
  cutilSafeCall(cudaFree(expKx));
  cutilSafeCall(cudaFree(expKy));
  cutilSafeCall(cudaFree(expKz));


  cutilSafeCall(cudaFree(vxZ));
  cutilSafeCall(cudaFree(vyZ));
  cutilSafeCall(cudaFree(vzZ));



  cout << "FREE MEMORY GPU :               DONE" << endl; 


  return 1;
}





