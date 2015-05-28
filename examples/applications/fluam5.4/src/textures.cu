// Filename: textures.cu
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


void initTextures(){
  //texvecino0GPU
  texvecino0GPU.normalized = false;
  texvecino0GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino0GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino0GPU,vecino0GPU,ncells*sizeof(int)));
  //texvecino1GPU
  texvecino1GPU.normalized = false;
  texvecino1GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino1GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino1GPU,vecino1GPU,ncells*sizeof(int)));
  //texvecino2GPU
  texvecino2GPU.normalized = false;
  texvecino2GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino2GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino2GPU,vecino2GPU,ncells*sizeof(int)));
  //texvecino3GPU
  texvecino3GPU.normalized = false;
  texvecino3GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino3GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino3GPU,vecino3GPU,ncells*sizeof(int)));
  //texvecino4GPU
  texvecino4GPU.normalized = false;
  texvecino4GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino4GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino4GPU,vecino4GPU,ncells*sizeof(int)));
  //texvecino5GPU
  texvecino5GPU.normalized = false;
  texvecino5GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino5GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino5GPU,vecino5GPU,ncells*sizeof(int)));
  //texvecinopxpyGPU
  texvecinopxpyGPU.normalized = false;
  texvecinopxpyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpyGPU,vecinopxpyGPU,ncells*sizeof(int)));
  //texvecinopxmyGPU
  texvecinopxmyGPU.normalized = false;
  texvecinopxmyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmyGPU,vecinopxmyGPU,ncells*sizeof(int)));
  //texvecinopxpzGPU
  texvecinopxpzGPU.normalized = false;
  texvecinopxpzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpzGPU,vecinopxpzGPU,ncells*sizeof(int)));
  //texvecinopxmzGPU
  texvecinopxmzGPU.normalized = false;
  texvecinopxmzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmzGPU,vecinopxmzGPU,ncells*sizeof(int)));
  //texvecinomxpyGPU
  texvecinomxpyGPU.normalized = false;
  texvecinomxpyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpyGPU,vecinomxpyGPU,ncells*sizeof(int)));
  //texvecinomxmyGPU
  texvecinomxmyGPU.normalized = false;
  texvecinomxmyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmyGPU,vecinomxmyGPU,ncells*sizeof(int)));
  //texvecinomxpzGPU
  texvecinomxpzGPU.normalized = false;
  texvecinomxpzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpzGPU,vecinomxpzGPU,ncells*sizeof(int)));
  //texvecinomxmzGPU
  texvecinomxmzGPU.normalized = false;
  texvecinomxmzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmzGPU,vecinomxmzGPU,ncells*sizeof(int)));
  //texvecinopypzGPU
  texvecinopypzGPU.normalized = false;
  texvecinopypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopypzGPU,vecinopypzGPU,ncells*sizeof(int)));
  //texvecinopymzGPU
  texvecinopymzGPU.normalized = false;
  texvecinopymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopymzGPU,vecinopymzGPU,ncells*sizeof(int)));
  //texvecinomypzGPU
  texvecinomypzGPU.normalized = false;
  texvecinomypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomypzGPU,vecinomypzGPU,ncells*sizeof(int)));
  //texvecinomymzGPU
  texvecinomymzGPU.normalized = false;
  texvecinomymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomymzGPU,vecinomymzGPU,ncells*sizeof(int)));
  //texvecinopxpypzGPU
  texvecinopxpypzGPU.normalized = false;
  texvecinopxpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpypzGPU,vecinopxpypzGPU,ncells*sizeof(int)));
  //texvecinopxpymzGPU
  texvecinopxpymzGPU.normalized = false;
  texvecinopxpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpymzGPU,vecinopxpymzGPU,ncells*sizeof(int)));
  //texvecinopxmypzGPU
  texvecinopxmypzGPU.normalized = false;
  texvecinopxmypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmypzGPU,vecinopxmypzGPU,ncells*sizeof(int)));
  //texvecinopxmymzGPU
  texvecinopxmymzGPU.normalized = false;
  texvecinopxmymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmymzGPU,vecinopxmymzGPU,ncells*sizeof(int)));
  //texvecinomxpypzGPU
  texvecinomxpypzGPU.normalized = false;
  texvecinomxpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpypzGPU,vecinomxpypzGPU,ncells*sizeof(int)));
  //texvecinomymzGPU
  texvecinomxpymzGPU.normalized = false;
  texvecinomxpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpymzGPU,vecinomxpymzGPU,ncells*sizeof(int)));
  //texvecinomxmypzGPU
  texvecinomxmypzGPU.normalized = false;
  texvecinomxmypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmypzGPU,vecinomxmypzGPU,ncells*sizeof(int)));
  //texvecinomxmymzGPU
  texvecinomxmymzGPU.normalized = false;
  texvecinomxmymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmymzGPU,vecinomxmymzGPU,ncells*sizeof(int)));


  //texVxGPU;
  texVxGPU.normalized = false;
  texVxGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texVxGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncells*sizeof(double)));
  //texVyGPU;
  texVyGPU.normalized = false;
  texVyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texVyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncells*sizeof(double)));
  //texVzGPU;
  texVzGPU.normalized = false;
  texVzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texVzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncells*sizeof(double)));


  if((setboundary==1) || (setparticles==1)){
    //texrxboundaryGPU
    texrxboundaryGPU.normalized = false;
    texrxboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texrxboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
    //texryboundaryGPU
    texryboundaryGPU.normalized = false;
    texryboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texryboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
    //texrzboundaryGPU
    texrzboundaryGPU.normalized = false;
    texrzboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texrzboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));
    //texCountParticlesInCellX;
    texCountParticlesInCellX.normalized = false;
    texCountParticlesInCellX.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texCountParticlesInCellX.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texCountParticlesInCellX,countparticlesincellX,ncells*sizeof(int)));
    //texCountParticlesInCellY;
    texCountParticlesInCellY.normalized = false;
    texCountParticlesInCellY.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texCountParticlesInCellY.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texCountParticlesInCellY,countparticlesincellY,ncells*sizeof(int)));
    //texCountParticlesInCellZ;
    texCountParticlesInCellZ.normalized = false;
    texCountParticlesInCellZ.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texCountParticlesInCellZ.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texCountParticlesInCellZ,countparticlesincellZ,ncells*sizeof(int)));
    //texPartInCellX;
    texPartInCellX.normalized = false;
    texPartInCellX.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texPartInCellX.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texPartInCellX,partincellX,maxNumberPartInCell*ncells*sizeof(int)));
    //texPartInCellY;
    texPartInCellY.normalized = false;
    texPartInCellY.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texPartInCellY.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texPartInCellY,partincellY,maxNumberPartInCell*ncells*sizeof(int)));
    //texPartInCellZ;
    texPartInCellZ.normalized = false;
    texPartInCellZ.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texPartInCellZ.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texPartInCellZ,partincellZ,maxNumberPartInCell*ncells*sizeof(int)));


    /*//texfxboundaryGPU
    texfxboundaryGPU.normalized = false;
    texfxboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texfxboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texfxboundaryGPU,fxboundaryGPU,27*nboundary*sizeof(double)));
    //texfyboundaryGPU
    texfyboundaryGPU.normalized = false;
    texfyboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texfyboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texfyboundaryGPU,fyboundaryGPU,27*nboundary*sizeof(double)));
    //texfzboundaryGPU
    texfzboundaryGPU.normalized = false;
    texfzboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texfzboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texfzboundaryGPU,fzboundaryGPU,27*nboundary*sizeof(double)));*/
  }

  if(setparticles==1){
    //texCountParticlesInCellNonBonded;
    texCountParticlesInCellNonBonded.normalized = false;
    texCountParticlesInCellNonBonded.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texCountParticlesInCellNonBonded.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texCountParticlesInCellNonBonded,countPartInCellNonBonded,numNeighbors*sizeof(int)));
    //texPartInCellNonBonded;
    texPartInCellNonBonded.normalized = false;
    texPartInCellNonBonded.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texPartInCellNonBonded.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texPartInCellNonBonded,partInCellNonBonded,
				   maxNumberPartInCellNonBonded*numNeighbors*sizeof(int)));
    //texneighbor0GPU
    texneighbor0GPU.normalized = false;
    texneighbor0GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbor0GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbor0GPU,neighbor0GPU,numNeighbors*sizeof(int)));
    //texneighbor1GPU
    texneighbor1GPU.normalized = false;
    texneighbor1GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbor1GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbor1GPU,neighbor1GPU,numNeighbors*sizeof(int)));
    //texneighbor2GPU
    texneighbor2GPU.normalized = false;
    texneighbor2GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbor2GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbor2GPU,neighbor2GPU,numNeighbors*sizeof(int)));
    //texneighbor3GPU
    texneighbor3GPU.normalized = false;
    texneighbor3GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbor3GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbor3GPU,neighbor3GPU,numNeighbors*sizeof(int)));
    //texneighbor4GPU
    texneighbor4GPU.normalized = false;
    texneighbor4GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbor4GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbor4GPU,neighbor4GPU,numNeighbors*sizeof(int)));
    //texneighbor5GPU
    texneighbor5GPU.normalized = false;
    texneighbor5GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbor5GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbor5GPU,neighbor5GPU,numNeighbors*sizeof(int)));
    //texneighborpxpyGPU
    texneighborpxpyGPU.normalized = false;
    texneighborpxpyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxpyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxpyGPU,neighborpxpyGPU,numNeighbors*sizeof(int)));
    //texneighborpxmyGPU
    texneighborpxmyGPU.normalized = false;
    texneighborpxmyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxmyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxmyGPU,neighborpxmyGPU,numNeighbors*sizeof(int)));
    //texneighborpxpzGPU
    texneighborpxpzGPU.normalized = false;
    texneighborpxpzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxpzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxpzGPU,neighborpxpzGPU,numNeighbors*sizeof(int)));
    //texneighborpxmzGPU
    texneighborpxmzGPU.normalized = false;
    texneighborpxmzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxmzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxmzGPU,neighborpxmzGPU,numNeighbors*sizeof(int)));
    //texneighbormxpyGPU
    texneighbormxpyGPU.normalized = false;
    texneighbormxpyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxpyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxpyGPU,neighbormxpyGPU,numNeighbors*sizeof(int)));
    //texneighbormxmyGPU
    texneighbormxmyGPU.normalized = false;
    texneighbormxmyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxmyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxmyGPU,neighbormxmyGPU,numNeighbors*sizeof(int)));
    //texneighbormxpzGPU
    texneighbormxpzGPU.normalized = false;
    texneighbormxpzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxpzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxpzGPU,neighbormxpzGPU,numNeighbors*sizeof(int)));
    //texneighbormxmzGPU
    texneighbormxmzGPU.normalized = false;
    texneighbormxmzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxmzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxmzGPU,neighbormxmzGPU,numNeighbors*sizeof(int)));
    //texneighborpypzGPU
    texneighborpypzGPU.normalized = false;
    texneighborpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpypzGPU,neighborpypzGPU,numNeighbors*sizeof(int)));
    //texneighborpymzGPU
    texneighborpymzGPU.normalized = false;
    texneighborpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpymzGPU,neighborpymzGPU,numNeighbors*sizeof(int)));
    //texneighbormypzGPU
    texneighbormypzGPU.normalized = false;
    texneighbormypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormypzGPU,neighbormypzGPU,numNeighbors*sizeof(int)));
    //texneighbormymzGPU
    texneighbormymzGPU.normalized = false;
    texneighbormymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormymzGPU,neighbormymzGPU,numNeighbors*sizeof(int)));
    //texneighborpxpypzGPU
    texneighborpxpypzGPU.normalized = false;
    texneighborpxpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxpypzGPU,neighborpxpypzGPU,numNeighbors*sizeof(int)));
    //texneighborpxpymzGPU
    texneighborpxpymzGPU.normalized = false;
    texneighborpxpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxpymzGPU,neighborpxpymzGPU,numNeighbors*sizeof(int)));
    //texneighborpxmypzGPU
    texneighborpxmypzGPU.normalized = false;
    texneighborpxmypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxmypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxmypzGPU,neighborpxmypzGPU,numNeighbors*sizeof(int)));
    //texneighborpxmymzGPU
    texneighborpxmymzGPU.normalized = false;
    texneighborpxmymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighborpxmymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighborpxmymzGPU,neighborpxmymzGPU,numNeighbors*sizeof(int)));
    //texneighbormxpypzGPU
    texneighbormxpypzGPU.normalized = false;
    texneighbormxpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxpypzGPU,neighbormxpypzGPU,numNeighbors*sizeof(int)));
    //texneighbormymzGPU
    texneighbormxpymzGPU.normalized = false;
    texneighbormxpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxpymzGPU,neighbormxpymzGPU,numNeighbors*sizeof(int)));
    //texneighbormxmypzGPU
    texneighbormxmypzGPU.normalized = false;
    texneighbormxmypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxmypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxmypzGPU,neighbormxmypzGPU,numNeighbors*sizeof(int)));
    //texneighbormxmymzGPU
    texneighbormxmymzGPU.normalized = false;
    texneighbormxmymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texneighbormxmymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texneighbormxmymzGPU,neighbormxmymzGPU,numNeighbors*sizeof(int)));
    
    init_force_non_bonded();

  }

  cout << "TEXTURES BINDED " << endl;

}
