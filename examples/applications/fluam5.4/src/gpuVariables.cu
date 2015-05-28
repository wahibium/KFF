// Filename: gpuVariables.cu
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



/*********************************************************/
/* CELL VARIABLES FOR GPU                                */
/*********************************************************/ 
//DATA FOR RANDOM: 90.000.000 INT = 343 MB
//DATA FOR EACH CELL: 26 INT + 10 DOUBLE = 144 B
//DATA FOR EACH BOUNDARY: 1 INT + 34 DOUBLE = 140 B

typedef struct{
  int* vecino0GPU;
  int* vecino1GPU;
  int* vecino2GPU;
  int* vecino3GPU;
  int* vecino4GPU;
  int* vecino5GPU;
  int* vecinopxpyGPU;
  int* vecinopxmyGPU;
  int* vecinopxpzGPU;
  int* vecinopxmzGPU;
  int* vecinomxpyGPU;
  int* vecinomxmyGPU;
  int* vecinomxpzGPU;
  int* vecinomxmzGPU;
  int* vecinopypzGPU;
  int* vecinopymzGPU;
  int* vecinomypzGPU;
  int* vecinomymzGPU;
  int* vecinopxpypzGPU;
  int* vecinopxpymzGPU;
  int* vecinopxmypzGPU;
  int* vecinopxmymzGPU;
  int* vecinomxpypzGPU;
  int* vecinomxpymzGPU;
  int* vecinomxmypzGPU;
  int* vecinomxmymzGPU;
} vecinos;

typedef struct{
  double* fcell;
  double* fvec0;
  double* fvec1;
  double* fvec2;
  double* fvec3;
  double* fvec4;
  double* fvec5;
  double* fvecpxpy;
  double* fvecpxmy;
  double* fvecpxpz;
  double* fvecpxmz;
  double* fvecmxpy;
  double* fvecmxmy;
  double* fvecmxpz;
  double* fvecmxmz;
  double* fvecpypz;
  double* fvecpymz;
  double* fvecmypz;
  double* fvecmymz;
  double* fvecpxpypz;
  double* fvecpxpymz;
  double* fvecpxmypz;
  double* fvecpxmymz;
  double* fvecmxpypz;
  double* fvecmxpymz;
  double* fvecmxmypz;
  double* fvecmxmymz;
  int* position;
} fvec;

typedef struct{
  int* countparticlesincellX;
  int* countparticlesincellY;
  int* countparticlesincellZ;
  int* partincellX;
  int* partincellY;
  int* partincellZ;
  int* countPartInCellNonBonded;
  int* partInCellNonBonded;
  //freeEnergyCompressibleParticles
  int* countparticlesincell;
  int* partincell;
} particlesincell;

typedef struct{
  cufftDoubleComplex* gradKx;
  cufftDoubleComplex* gradKy;
  cufftDoubleComplex* gradKz;
  cufftDoubleComplex* expKx;
  cufftDoubleComplex* expKy;
  cufftDoubleComplex* expKz;
} prefactorsFourier;


__constant__ int mxGPU, myGPU, mzGPU;
__constant__ int mxtGPU, mytGPU, mztGPU, mxmytGPU;
__constant__ int ncellsGPU, ncellstGPU;
__constant__ bool thermostatGPU;
__constant__ double lxGPU, lyGPU, lzGPU;
__constant__ double velocityboundaryGPU;
__constant__ double dtGPU;
__constant__ int numberneighboursGPU;
//int *cellneighbourGPU;// cellneighbourGPU[i*numberneighbours+j] neighbour j cell i
__constant__ double volumeGPU;
__constant__ double exGPU[6], eyGPU[6], ezGPU[6];
__constant__ double dxGPU, dyGPU, dzGPU; 
__constant__ double invdxGPU, invdyGPU, invdzGPU; 
__constant__ double invlxGPU, invlyGPU, invlzGPU;
__constant__ double invdtGPU;

//__device__ double *massGPU;
__device__ double *densityGPU;
__device__ double *densityPredictionGPU;
__device__ double *vxGPU, *vyGPU, *vzGPU;
__device__ double *vxPredictionGPU, *vyPredictionGPU, *vzPredictionGPU;
__device__ double *fxGPU, *fyGPU, *fzGPU;
__device__ double *dmGPU;
__device__ double *dpxGPU, *dpyGPU, *dpzGPU;
__device__ double *rxcellGPU, *rycellGPU, *rzcellGPU;
__device__ double *advXGPU, *advYGPU, *advZGPU;
__device__ double *omegaGPU;

//IMEXRK
__device__ double *vx2GPU, *vy2GPU, *vz2GPU;
__device__ double *vx3GPU, *vy3GPU, *vz3GPU;
__device__ double *rxboundary2GPU, *ryboundary2GPU, *rzboundary2GPU;
__device__ double *rxboundary3GPU, *ryboundary3GPU, *rzboundary3GPU;
__device__ double *vxboundary2GPU, *vyboundary2GPU, *vzboundary2GPU;
__device__ double *vxboundary3GPU, *vyboundary3GPU, *vzboundary3GPU;
__device__ double *fx2GPU, *fy2GPU, *fz2GPU;
__device__ double *fx3GPU, *fy3GPU, *fz3GPU;

//__constant__ double omega1, omega2, omega3, omega4, omega5;


//Binary Mixture
__device__ double *cGPU, *cPredictionGPU, *dcGPU;

__constant__ double cWall0GPU, cWall1GPU, densityWall0GPU, densityWall1GPU;
__constant__ double vxWall0GPU, vxWall1GPU;
__constant__ double vyWall0GPU, vyWall1GPU;
__constant__ double vzWall0GPU, vzWall1GPU;
__constant__ double diffusionGPU, massSpecies0GPU, massSpecies1GPU;
__constant__ double shearviscosityGPU;
__constant__ double bulkviscosityGPU;
__constant__ double temperatureGPU;
__constant__ double pressurea0GPU;
__constant__ double pressurea1GPU;
__constant__ double pressurea2GPU;
__constant__ double densfluidGPU;

__constant__ double fact1GPU, fact2GPU, fact3GPU, fact4GPU;
__constant__ double fact5GPU, fact6GPU, fact7GPU;
__constant__ double volumeboundaryconstGPU;

__constant__ double soretCoefficientGPU, gradTemperatureGPU;

__constant__ double extraMobilityGPU;
__constant__ bool setExtraMobilityGPU;

__device__ double *rxboundaryGPU, *ryboundaryGPU, *rzboundaryGPU;
__device__ double *vxboundaryGPU, *vyboundaryGPU, *vzboundaryGPU;
__device__ double *fxboundaryGPU, *fyboundaryGPU, *fzboundaryGPU;
__device__ double *fboundaryOmega;
__device__ double *volumeboundaryGPU;
__device__ double *fbcell;

//__device__ double *rxParticleGPU, *ryParticleGPU, *rzParticleGPU;
//__device__ double *vxParticleGPU, *vyParticleGPU, *vzParticleGPU;
__constant__ double massParticleGPU, volumeParticleGPU;
__constant__ int npGPU;
__constant__ bool setparticlesGPU, setboundaryGPU;
__constant__ double omega0GPU;
/*__device__ double *fb0, *fb1, *fb2, *fb3, *fb4, *fb5;
__device__ double *fbpxpy, *fbpxmy, *fbpxpz, *fbpxmz;
__device__ double *fbmxpy, *fbmxmy, *fbmxpz, *fbmxmz;
__device__ double *fbpypz, *fbpymz, *fbmypz, *fbmymz;
__device__ double *fbpxpypz, *fbpxpymz, *fbpxmypz, *fbpxmymz;
__device__ double *fbmxpypz, *fbmxpymz, *fbmxmypz, *fbmxmymz;
__device__ int *bposition;*/

__device__ int *ghostIndexGPU, *realIndexGPU;
__device__ int *ghostToPIGPU, *ghostToGhostGPU;
__device__ int *vecino0GPU, *vecino1GPU, *vecino2GPU;
__device__ int *vecino3GPU, *vecino4GPU, *vecino5GPU;
__device__ int *vecinopxpyGPU, *vecinopxmyGPU, *vecinopxpzGPU, *vecinopxmzGPU;
__device__ int *vecinomxpyGPU, *vecinomxmyGPU, *vecinomxpzGPU, *vecinomxmzGPU;
__device__ int *vecinopypzGPU, *vecinopymzGPU, *vecinomypzGPU, *vecinomymzGPU;
__device__ int *vecinopxpypzGPU, *vecinopxpymzGPU, *vecinopxmypzGPU, *vecinopxmymzGPU;
__device__ int *vecinomxpypzGPU, *vecinomxpymzGPU, *vecinomxmypzGPU, *vecinomxmymzGPU; 
__constant__ int nboundaryGPU;
__constant__ double vboundaryGPU;

__device__ int *neighbor0GPU, *neighbor1GPU, *neighbor2GPU;
__device__ int *neighbor3GPU, *neighbor4GPU, *neighbor5GPU;
__device__ int *neighborpxpyGPU, *neighborpxmyGPU, *neighborpxpzGPU, *neighborpxmzGPU;
__device__ int *neighbormxpyGPU, *neighbormxmyGPU, *neighbormxpzGPU, *neighbormxmzGPU;
__device__ int *neighborpypzGPU, *neighborpymzGPU, *neighbormypzGPU, *neighbormymzGPU;
__device__ int *neighborpxpypzGPU, *neighborpxpymzGPU, *neighborpxmypzGPU, *neighborpxmymzGPU;
__device__ int *neighbormxpypzGPU, *neighbormxpymzGPU, *neighbormxmypzGPU, *neighbormxmymzGPU;
__constant__ int mxNeighborsGPU, myNeighborsGPU, mzNeighborsGPU, mNeighborsGPU;


__device__ int *partincellX, *partincellY, *partincellZ;
__device__ int *countparticlesincellX, *countparticlesincellY, *countparticlesincellZ;
__device__ int *partInCellNonBonded, *countPartInCellNonBonded;
__device__ int *countparticlesincell, *partincell;
__constant__ int maxNumberPartInCellGPU, maxNumberPartInCellNonBondedGPU;
__device__ int *errorKernel;
__constant__ double cutoffGPU, invcutoffGPU, invcutoff2GPU;

__constant__ double *saveForceX, *saveForceY, *saveForceZ;

//WAVE SOURCE
__device__ long long *stepGPU;
__constant__ double densityConstGPU, dDensityGPU;

__device__ vecinos *vec;
__device__ fvec *fb;
__device__ particlesincell *pc;

__device__ double *rxCheckGPU, *ryCheckGPU, *rzCheckGPU;
__device__ double *vxCheckGPU, *vyCheckGPU, *vzCheckGPU;

cudaArray *cuArrayDelta;
cudaArray *cuArrayDeltaDerived;
cudaArray *forceNonBonded1;

texture<int, 1> texvecino0GPU;
texture<int, 1> texvecino1GPU;
texture<int, 1> texvecino2GPU;
texture<int, 1> texvecino3GPU;
texture<int, 1> texvecino4GPU;
texture<int, 1> texvecino5GPU;
texture<int, 1> texvecinopxpyGPU;
texture<int, 1> texvecinopxmyGPU;
texture<int, 1> texvecinopxpzGPU;
texture<int, 1> texvecinopxmzGPU;
texture<int, 1> texvecinomxpyGPU;
texture<int, 1> texvecinomxmyGPU;
texture<int, 1> texvecinomxpzGPU;
texture<int, 1> texvecinomxmzGPU;
texture<int, 1> texvecinopypzGPU;
texture<int, 1> texvecinopymzGPU;
texture<int, 1> texvecinomypzGPU;
texture<int, 1> texvecinomymzGPU;
texture<int, 1> texvecinopxpypzGPU;
texture<int, 1> texvecinopxpymzGPU;
texture<int, 1> texvecinopxmypzGPU;
texture<int, 1> texvecinopxmymzGPU;
texture<int, 1> texvecinomxpypzGPU;
texture<int, 1> texvecinomxpymzGPU;
texture<int, 1> texvecinomxmypzGPU;
texture<int, 1> texvecinomxmymzGPU;

texture<int2, 1> texrxboundaryGPU;
texture<int2, 1> texryboundaryGPU;
texture<int2, 1> texrzboundaryGPU;
texture<int2, 1> texfxboundaryGPU;
texture<int2, 1> texfyboundaryGPU;
texture<int2, 1> texfzboundaryGPU;

cudaArray *cuArrayDeltaPBC;

texture<int, 1> texCountParticlesInCellX;
texture<int, 1> texCountParticlesInCellY;
texture<int, 1> texCountParticlesInCellZ;
texture<int, 1> texPartInCellX;
texture<int, 1> texPartInCellY;
texture<int, 1> texPartInCellZ;

texture<int2, 1> texVxGPU;
texture<int2, 1> texVyGPU;
texture<int2, 1> texVzGPU;

texture<int, 1> texCountParticlesInCellNonBonded;
texture<int, 1> texPartInCellNonBonded;

texture<float, 1, cudaReadModeElementType> texforceNonBonded1;

texture<int, 1> texneighbor0GPU;
texture<int, 1> texneighbor1GPU;
texture<int, 1> texneighbor2GPU;
texture<int, 1> texneighbor3GPU;
texture<int, 1> texneighbor4GPU;
texture<int, 1> texneighbor5GPU;
texture<int, 1> texneighborpxpyGPU;
texture<int, 1> texneighborpxmyGPU;
texture<int, 1> texneighborpxpzGPU;
texture<int, 1> texneighborpxmzGPU;
texture<int, 1> texneighbormxpyGPU;
texture<int, 1> texneighbormxmyGPU;
texture<int, 1> texneighbormxpzGPU;
texture<int, 1> texneighbormxmzGPU;
texture<int, 1> texneighborpypzGPU;
texture<int, 1> texneighborpymzGPU;
texture<int, 1> texneighbormypzGPU;
texture<int, 1> texneighbormymzGPU;
texture<int, 1> texneighborpxpypzGPU;
texture<int, 1> texneighborpxpymzGPU;
texture<int, 1> texneighborpxmypzGPU;
texture<int, 1> texneighborpxmymzGPU;
texture<int, 1> texneighbormxpypzGPU;
texture<int, 1> texneighbormxpymzGPU;
texture<int, 1> texneighbormxmypzGPU;
texture<int, 1> texneighbormxmymzGPU;


//Incompressible
__device__ cufftDoubleComplex *WxZ, *WyZ, *WzZ;
__device__ cufftDoubleComplex *vxZ, *vyZ, *vzZ;
__device__ cufftDoubleComplex *cZ;
__device__ cufftDoubleComplex *gradKx, *gradKy, *gradKz;
__device__ cufftDoubleComplex *expKx, *expKy, *expKz;
__device__ prefactorsFourier *pF;


//IncompressibleBoundaryRK2
__device__ double *rxboundaryPredictionGPU, *ryboundaryPredictionGPU, *rzboundaryPredictionGPU;
__device__ double *vxboundaryPredictionGPU, *vyboundaryPredictionGPU, *vzboundaryPredictionGPU;



//NEW Bonded forces
typedef struct{
  int *bondsParticleParticleGPU;
  int *bondsParticleParticleOffsetGPU;
  int *bondsIndexParticleParticleGPU;
  double *r0ParticleParticleGPU;
  double *kSpringParticleParticleGPU;


  int *bondsParticleFixedPointGPU;
  int *bondsParticleFixedPointOffsetGPU;
  //int *bondsIndexParticleFixedPointGPU;
  double *r0ParticleFixedPointGPU;
  double *kSpringParticleFixedPointGPU;
  double *rxFixedPointGPU;
  double *ryFixedPointGPU;
  double *rzFixedPointGPU;
} bondedForcesVariables;

__constant__ bool bondedForcesGPU;
__device__ bondedForcesVariables *bFV;
__device__ int *bondsParticleParticleGPU;
__device__ int *bondsParticleParticleOffsetGPU;
__device__ int *bondsIndexParticleParticleGPU;
__device__ double *r0ParticleParticleGPU;
__device__ double *kSpringParticleParticleGPU;


__device__ int *bondsParticleFixedPointGPU;
__device__ int *bondsParticleFixedPointOffsetGPU;
//__device__ int *bondsIndexParticleFixedPointGPU;
__device__ double *r0ParticleFixedPointGPU;
__device__ double *kSpringParticleFixedPointGPU;
__device__ double *rxFixedPointGPU;
__device__ double *ryFixedPointGPU;
__device__ double *rzFixedPointGPU;
__constant__ bool particlesWallGPU;
__constant__ bool computeNonBondedForcesGPU;

