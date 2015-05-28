// Filename: freeCellsBinaryMixtureGPU.cu
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


bool freeCellsBinaryMixtureGPU(){
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

  cutilSafeCall(cudaFree(cGPU));
  cutilSafeCall(cudaFree(cPredictionGPU));
  cutilSafeCall(cudaFree(dcGPU));

  cutilSafeCall(cudaFree(dmGPU));
  cutilSafeCall(cudaFree(dpxGPU));
  cutilSafeCall(cudaFree(dpyGPU));
  cutilSafeCall(cudaFree(dpzGPU));

  cutilSafeCall(cudaFree(rxcellGPU));
  cutilSafeCall(cudaFree(rycellGPU));
  cutilSafeCall(cudaFree(rzcellGPU));

  cutilSafeCall(cudaFree(ghostIndexGPU));
  cutilSafeCall(cudaFree(realIndexGPU));
  cutilSafeCall(cudaFree(ghostToPIGPU));
  cutilSafeCall(cudaFree(ghostToGhostGPU));

  cutilSafeCall(cudaFree(stepGPU));

  cout << "FREE MEMORY GPU :               DONE" << endl; 


  return 1;
}

