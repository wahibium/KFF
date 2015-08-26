// Filename: createBoundariesRK2GPU.cu
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


bool createBoundaries4ptGPU(){

  if(setparticles==0) np=0;
  
  cudaMemcpyToSymbol(nboundaryGPU,&nboundary,sizeof(int));
  cudaMemcpyToSymbol(npGPU,&np,sizeof(int));
  cutilSafeCall(cudaMemcpyToSymbol(maxNumberPartInCellGPU,&maxNumberPartInCell,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(maxNumberPartInCellNonBondedGPU,&maxNumberPartInCellNonBonded,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(particlesWallGPU,&particlesWall,sizeof(bool)));
  cudaMemcpyToSymbol(computeNonBondedForcesGPU,&computeNonBondedForces,sizeof(bool));

  //Create boundaries and particles variables
  cutilSafeCall(cudaMalloc((void**)&rxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&ryboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&fxboundaryGPU,64*(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&fyboundaryGPU,64*(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&fzboundaryGPU,64*(nboundary+np)*sizeof(double)));

  //Initialize boundaries variables
  cutilSafeCall(cudaMemcpy(rxboundaryGPU,rxboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(ryboundaryGPU,ryboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(rzboundaryGPU,rzboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vxboundaryGPU,vxboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vyboundaryGPU,vyboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vzboundaryGPU,vzboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));

  //Initialize particles variables
  cutilSafeCall(cudaMemcpy(&rxboundaryGPU[nboundary],rxParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&ryboundaryGPU[nboundary],ryParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&rzboundaryGPU[nboundary],rzParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&vxboundaryGPU[nboundary],vxParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&vyboundaryGPU[nboundary],vyParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&vzboundaryGPU[nboundary],vzParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  
  //Copy some constants
  cutilSafeCall(cudaMemcpyToSymbol(volumeboundaryconstGPU,&volumeboundaryconst,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(massParticleGPU,&mass,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(volumeParticleGPU,&volumeParticle,sizeof(double)));
 
              
  cutilSafeCall(cudaMalloc((void**)&countparticlesincellX,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&countparticlesincellY,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&countparticlesincellZ,ncells*sizeof(int)));
  int aux[ncells];
  for(int i=0;i<ncells;i++) aux[i] = 0;
  cudaMemcpy(countparticlesincellX,aux,ncells*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(countparticlesincellY,aux,ncells*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(countparticlesincellZ,aux,ncells*sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&partincellX,maxNumberPartInCell*ncells*sizeof(int));
  cudaMalloc((void**)&partincellY,maxNumberPartInCell*ncells*sizeof(int));
  cudaMalloc((void**)&partincellZ,maxNumberPartInCell*ncells*sizeof(int));
  

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

  if(setparticles){
    int mxPart = int(lx/cutoff);
    if(mxPart < 3) mxPart = 3;
    int myPart = int(ly/cutoff);
    if(myPart < 3) myPart = 3;
    int mzPart = int(lz/cutoff);
    if(mzPart < 3) mzPart = 3;
    numNeighbors = mxPart * myPart * mzPart;

    cutilSafeCall(cudaMalloc((void**)&countPartInCellNonBonded,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&partInCellNonBonded,
			     maxNumberPartInCellNonBonded*numNeighbors*sizeof(int)));

    cutilSafeCall(cudaMemcpyToSymbol(mxNeighborsGPU,&mxPart,sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(myNeighborsGPU,&myPart,sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(mzNeighborsGPU,&mzPart,sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(mNeighborsGPU,&numNeighbors,sizeof(int)));
    
    cutilSafeCall(cudaMalloc((void**)&neighbor0GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor1GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor2GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor3GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor4GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor5GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmymzGPU,numNeighbors*sizeof(int)));
    
    cutilSafeCall(cudaMemcpyToSymbol(cutoffGPU,&cutoff,sizeof(double)));
    double invcutoff = 1./cutoff;
    cutilSafeCall(cudaMemcpyToSymbol(invcutoffGPU,&invcutoff,sizeof(double)));
    invcutoff = 1./(cutoff * cutoff);
    cutilSafeCall(cudaMemcpyToSymbol(invcutoff2GPU,&invcutoff,sizeof(double)));
    
    
    //texCountParticlesInCellNonBonded;
    texCountParticlesInCellNonBonded.normalized = false;
    texCountParticlesInCellNonBonded.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
    texCountParticlesInCellNonBonded.filterMode = 
      cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
    cutilSafeCall( cudaBindTexture(0,texCountParticlesInCellNonBonded,
				   countPartInCellNonBonded,numNeighbors*sizeof(int)));
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
    
    //init_force_non_bonded();

    int block = (numNeighbors-1)/128 + 1;

    initializeNeighbors<<<block,128>>>(neighbor1GPU,neighbor2GPU,neighbor3GPU,neighbor4GPU,
				       neighborpxpyGPU,neighborpxmyGPU,neighborpxpzGPU,neighborpxmzGPU,
				       neighbormxpyGPU,neighbormxmyGPU,neighbormxpzGPU,neighbormxmzGPU,
				       neighborpypzGPU,neighborpymzGPU,neighbormypzGPU,neighbormymzGPU,
				       neighborpxpypzGPU,neighborpxpymzGPU,
				       neighborpxmypzGPU,neighborpxmymzGPU,
				       neighbormxpypzGPU,neighbormxpymzGPU,
				       neighbormxmypzGPU,neighbormxmymzGPU);
    initializeNeighbors2<<<block,128>>>(neighbor0GPU,neighbor1GPU,neighbor2GPU,
					neighbor3GPU,neighbor4GPU,neighbor5GPU);

    initForcesNonBonded();

  }


  initDelta();
  allocateErrorArray();

  cutilSafeCall(cudaMalloc((void**)&pc,sizeof(particlesincell)));
  
  initParticlesInCell<<<1,1>>>(partincellX,partincellY,partincellZ,
			       countparticlesincellX,countparticlesincellY,countparticlesincellZ,
			       countPartInCellNonBonded,partInCellNonBonded,pc);

  //No-slip Test
  //cutilSafeCall(cudaMalloc((void**)&saveForceX,np*sizeof(double)));
  //cutilSafeCall(cudaMalloc((void**)&saveForceY,np*sizeof(double)));
  //cutilSafeCall(cudaMalloc((void**)&saveForceZ,np*sizeof(double)));
  

  double auxDouble[64*(nboundary+np)];
  for(int i=0;i<64*(nboundary+np);i++) auxDouble[i] = 0;
  cutilSafeCall(cudaMemcpy(fxboundaryGPU,auxDouble,64*(nboundary+np)*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(fyboundaryGPU,auxDouble,64*(nboundary+np)*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(fzboundaryGPU,auxDouble,64*(nboundary+np)*sizeof(double),cudaMemcpyHostToDevice));


  //Copy constant memory
  cudaMemcpyToSymbol(bondedForcesGPU,&bondedForces,sizeof(bool));

  
  cout << "CREATE BOUNDARIES GPU :         DONE" << endl; 

  return 1;
}
