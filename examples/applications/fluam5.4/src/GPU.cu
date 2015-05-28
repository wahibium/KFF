// Filename: GPU.cu
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


#include "header.h"
#include "cells.h"
#include "fluid.h"
#include "boundary.h"
#include "headerOtherFluidVariables.h"
#include "particles.h"
#include "hydroAnalysis.h"

#define cutilSafeCall(i) __cutilSafeCall(i, __FILE__, __LINE__)

inline void __cutilSafeCall(cudaError_t i, const char *file, const int line ){
  if(i!=cudaSuccess){
    printf("Error in %s at line %d with error code \"%s\"\n", file, line, cudaGetErrorString(i));
    exit(i);
  }  
  return;
}



//GPU staff
//#include <cutil_inline.h>
#include <cufft.h>
#include "curand.h"
#include "curand_kernel.h"
#include "gpuVariables.cu"
#include "moduGPU.cu"
#include "texturesCells.cu"
#include "initGhostIndexGPU.cu"
#include "pressureGPU.cu"


static __inline__ __device__ double fetch_double(texture<int2,1> t, int i){
  int2 v = tex1Dfetch(t,i);
  return __hiloint2double(v.y,v.x);
}

//schmeRK3
#include "initializeVecinosGPU.cu"
#include "createCellsGPU.cu"
#include "freeCellsGPU.cu"
//#include "freeMemoryRK3GPU.cu"
#include "initializeFluidGPU.cu"
#include "saveFunctionsSchemeRK3.cu"
#include "gpuToHostRK3.cu"
//#include "kernelFeedGhostCellsRK3.cu"
#include "kernelDpRK3.cu"
#include "runSchemeRK3.cu"

//schemeThermostat
#include "randomNumbers.cu"
#include "saveFunctionsSchemeThermostat.cu"
#include "kernelDpThermostat.cu"
#include "runSchemeThermostat.cu"

//schemeBoundary
#include "allocateErrorArray.cu"
#include "deltaGPU.cu"
#include "checkVelocity.cu"
#include "initParticlesInCell.cu"
#include "initializeNeighborsGPU.cu"
#include "createBoundariesGPU.cu"
#include "freeBoundariesGPU.cu"
#include "texturesBoundaries.cu"
#include "forceBoundaryGPU.cu"
#include "initForcesNonBonded.cu"

//NEW bonded forces
#include "createBondedForcesGPU.cu"
#include "freeBondedForcesGPU.cu"
#include "forceBondedGPU.cu"

#include "nonBondedForce.cu"
#include "nonBondedForceExtraPressure.cu" //Erase?
#include "updateFluid.cu"
#include "findNeighborParticles.cu"
#include "boundaryParticlesFunction.cu"
#include "saveFunctionsSchemeBoundary.cu"
#include "interpolateField.cu"
#include "gpuToHostParticles.cu"
#include "runSchemeBoundary.cu"

//schmeRK3Ghost
#include "texturesCellsGhost.cu"
#include "createCellsGhostGPU.cu"
#include "freeCellsGhostGPU.cu"
//#include "freeMemoryRK3GPU.cu"
#include "initializeFluidGhostGPU.cu"
#include "saveFunctionsSchemeRK3Ghost.cu"
#include "gpuToHostRK3Ghost.cu"
#include "kernelFeedGhostCellsRK3.cu"
#include "kernelDpRK3Ghost.cu"
#include "runSchemeRK3Ghost.cu"


//schemeBinaryMixture
#include "createCellsBinaryMixtureGPU.cu"
#include "freeCellsBinaryMixtureGPU.cu"
#include "initializeFluidBinaryMixtureGPU.cu"
#include "saveFunctionsSchemeBinaryMixture.cu"
#include "gpuToHostBinaryMixture.cu"
#include "diffusionBM.cu"
#include "kernelFeedGhostCellsBinaryMixture.cu"
#include "kernelDpBinaryMixture.cu"
#include "runSchemeBinaryMixture.cu"

//schemeBinaryMixtureWall: it needs the files of schemeBinaryMixture
#include "kernelFeedGhostCellsBinaryMixtureWall.cu"
#include "copyToGPUBinaryMixtureWall.cu"
#include "runSchemeBinaryMixtureWall.cu"

//schemeGiantFluctuations:
//it needs files from schemeBinaryMixture and schemeBinaryMixtureWall
#include "copyToGPUGiantFluctuations.cu"
#include "kernelFeedGhostCellsGiantFluctuations.cu"
#include "kernelDpGiantFluctuations.cu"
#include "runSchemeGiantFluctuations.cu"


//schemeContinuousGradient:
//it needs files from schemeBinaryMixture, schemeBinaryMixtureWall and
//schemeGiantFluctuations
//#include "kernelDpGiantFluctuations.cu"
#include "kernelDpContinuousGradient.cu"
#include "runSchemeContinuousGradient.cu"

//SchemeIncompressible
#include <cufft.h>
#include "projectionDivergenceFree.cu"
#include "saveFunctionsSchemeIncompressible.cu"
#include "createCellsIncompressibleGPU.cu"
#include "initializePrefactorFourierSpace.cu"
#include "initializeFluidIncompressibleGPU.cu"
#include "freeCellsIncompressibleGPU.cu"
#include "gpuToHostIncompressible.cu"
#include "kernelConstructW.cu"
#include "realToComplex.cu"
//#include "findNormalModes.cu"
#include "kernelUpdateVIncompressible.cu"
#include "runSchemeIncompressible.cu"

//SchemeIncompressibleBoundary
#include "forceIncompressibleBoundaryGPU.cu"
#include "gpuToHostIncompressibleBoundary.cu"
#include "nonBondedForceIncompressible.cu"
#include "boundaryParticlesFunctionIncompressible.cu"
#include "saveFunctionsSchemeIncompressibleBoundary.cu"
#include "runSchemeIncompressibleBoundary.cu"

//SchemeQuasiNeutrallyBuoyant
#include "createBoundariesRK2GPU.cu"
#include "freeBoundariesRK2GPU.cu"
#include "quasiNeutrallyBuoyantFunctions.cu"
#include "quasiNeutrallyBuoyantFunctions2.cu"
#include "calculateAdvectionFluid.cu"
#include "gpuToHostIncompressibleBoundaryRK2.cu"
#include "kernelSpreadParticles.cu"
#include "kernelConstructWQuasiNeutrallyBuoyant.cu"
#include "kernelCorrectionVQuasiNeutrallyBuoyant.cu"
#include "firstStepQuasiNeutrallyBuoyant.cu"
#include "runSchemeQuasiNeutrallyBuoyant.cu"

//SchemeQuasiNeutrallyBuoyant2D
#include "saveFunctionsSchemeIncompressibleBoundary2D.cu"
#include "createCellsIncompressible2DGPU.cu"
#include "quasiNeutrallyBuoyantFunctions2D.cu"
#include "firstStepQuasiNeutrallyBuoyant2D.cu"
#include "runSchemeQuasiNeutrallyBuoyant2D.cu"

//SchemeQuasiNeutrallyBuoyant4pt2D
//#include "createCellsIncompressible2DGPU.cu"
#include "quasiNeutrallyBuoyantFunctions4pt2D.cu"
#include "createBoundaries4ptGPU.cu"
#include "firstStepQuasiNeutrallyBuoyant4pt2D.cu"
#include "runSchemeQuasiNeutrallyBuoyant4pt2D.cu"


//SchemeCompressibleParticles
#include "calculateVelocityAtHalfTimeStepCompressibleParticles.cu"
#include "nonBondedForceCompressibleParticles.cu"
#include "nonBondedForceCompressibleParticlesExtraPressure.cu"
#include "boundaryParticlesFunctionCompressibleParticles.cu"
#include "kernelDpCompressibleParticles.cu"
#include "runSchemeCompressibleParticles.cu"


//SchemeIncompressibleBinaryMixture
#include "createCellsIncompressibleBinaryMixtureGPU.cu"
#include "freeCellsIncompressibleBinaryMixtureGPU.cu"
#include "kernelConstructWBinaryMixture.cu"
#include "kernelUpdateIncompressibleBinaryMixture.cu"
#include "saveFunctionsSchemeIncompressibleBinaryMixture.cu"
#include "runSchemeIncompressibleBinaryMixture.cu"

//SchemeIncompressibleBinaryMixtureMidPoint
#include "runSchemeIncompressibleBinaryMixtureMidPoint.cu"


//SchemeParticlesWall
#include "saveFunctionsSchemeParticlesWall.cu"
#include "initGhostIndexParticlesWallGPU.cu"
#include "kernelFeedGhostCellsParticlesWall.cu"
#include "kernelDpParticlesWall.cu"
#include "boundaryParticlesFunctionParticlesWall.cu"
#include "runSchemeParticlesWall.cu"


//SchemeTestJPS
#include "saveFunctionsSchemeTestJPS.cu"
#include "JPS.cu"
#include "runSchemeTestJPS.cu"


//SchemeFreeEnergyCompressibleParticles
#include "freeEnergyCompressibleParticles.cu"
#include "boundaryParticlesFunctionFreeEnergyCompressibleParticles.cu"
#include "kernelDpFreeEnergyCompressibleParticles.cu"
#include "runSchemeFreeEnergyCompressibleParticles.cu"

//SchemeSemiImplicitCompressibleParticles
#include "createCellsSemiImplicitCompressibleParticlesGPU.cu"
#include "freeCellsSemiImplicitCompressibleParticlesGPU.cu"
#include "kernelDpSemiImplicitCompressibleParticles.cu"
#include "kernelConstructWSemiImplicitCompressibleParticles.cu"
#include "kernelUpdateRhoSemiImplicit.cu"
#include "runSchemeSemiImplicitCompressibleParticles.cu"
#include "saveFunctionsSchemeSemiImplicitCompressibleParticles.cu"


//SchemeMomentumCoupling
#include "updateFluidMomentumCoupling.cu"
#include "nonBondedForceMomentumCoupling.cu"
#include "boundaryParticlesFunctionMomentumCoupling.cu"
#include "runSchemeMomentumCoupling.cu"

//SchemeStokesLimit
#include "createCellsStokesLimitGPU.cu"
#include "freeCellsStokesLimitGPU.cu"
#include "stokesLimitFunctions.cu"
#include "boundaryParticlesFunctionStokesLimit.cu"
#include "saveFunctionsSchemeStokesLimit.cu"
#include "gpuToHostStokesLimit.cu"
#include "runSchemeStokesLimit.cu"



