// Filename: runSchemeRK3.cu
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


bool runSchemeRK3(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;


  //Initialize textures cells
  if(!texturesCells()) return 0;
  
  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,vecino2GPU,vecino3GPU,vecino4GPU,
						      vecinopxpyGPU,vecinopxmyGPU,vecinopxpzGPU,vecinopxmzGPU,
						      vecinomxpyGPU,vecinomxmyGPU,vecinomxpzGPU,vecinomxmzGPU,
						      vecinopypzGPU,vecinopymzGPU,vecinomypzGPU,vecinomymzGPU,
						      vecinopxpypzGPU,vecinopxpymzGPU,vecinopxmypzGPU,vecinopxmymzGPU,
						      vecinomxpypzGPU,vecinomxpymzGPU,vecinomxmypzGPU,vecinomxmymzGPU);
  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,vecino1GPU,vecino2GPU,
						    vecino3GPU,vecino4GPU,vecino5GPU);


  step = -numstepsRelaxation;

  while(step<numsteps){
    //First substep RK3
    kernelDpRK3_1<<<numBlocks,threadsPerBlock>>>(densityGPU,densityGPU,
						 vxGPU,vyGPU,vzGPU,
						 dmGPU,
						 dpxGPU,dpyGPU,dpzGPU,
						 0,1);
    kernelDpRK3_2<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,
						 vxPredictionGPU,vyPredictionGPU,vzPredictionGPU,
						 dmGPU,
						 dpxGPU,dpyGPU,dpzGPU);
    cutilSafeCall( cudaBindTexture(0,texVxGPU,vxPredictionGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVyGPU,vyPredictionGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVzGPU,vzPredictionGPU,ncells*sizeof(double)));
    //Second substep RK3
    kernelDpRK3_1<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,densityGPU,
						 vxGPU,vyGPU,vzGPU,
						 dmGPU,
						 dpxGPU,dpyGPU,dpzGPU,
						 0.75,0.25);
    kernelDpRK3_2<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,
						 vxPredictionGPU,vyPredictionGPU,vzPredictionGPU,
						 dmGPU,
						 dpxGPU,dpyGPU,dpzGPU);
    //Third substep RK3
    kernelDpRK3_1<<<numBlocks,threadsPerBlock>>>(densityPredictionGPU,densityGPU,
						 vxGPU,vyGPU,vzGPU,
						 dmGPU,
						 dpxGPU,dpyGPU,dpzGPU,
						 1./3.,2./3.);
    kernelDpRK3_2<<<numBlocks,threadsPerBlock>>>(densityGPU,
						 vxGPU,vyGPU,vzGPU,
						 dmGPU,
						 dpxGPU,dpyGPU,dpzGPU);
    cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncells*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncells*sizeof(double)));
    
    
    step++;
    //cout << step << endl;

    if(!(step%samplefreq)&&(step>0)){
      cout << "RK3 " << step << endl;
      if(!gpuToHostRK3()) return 0;
      //if(!saveFunctionsSchemeRK3(1)) return 0;
    }
  }
  
    






  return 1;
}
