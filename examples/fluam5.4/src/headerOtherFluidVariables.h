// Filename: headerOtherFluidVariables.h
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


#ifdef GLOBALS_OTHER_FLUID_V
#define EXTERN_OTHER_FLUID_V 
#else
#define EXTERN_OTHER_FLUID_V extern
#endif


//Binary Mixture Begins
EXTERN_OTHER_FLUID_V bool setBinaryMixture;
EXTERN_OTHER_FLUID_V double diffusion, massSpecies0, massSpecies1;
EXTERN_OTHER_FLUID_V double concentration;//Mean concentration
EXTERN_OTHER_FLUID_V double *c;//Concentration for species 1
//__device__ double *cGPU, *cPredictionGPU, *dcGPU;
//__constant__ double diffusionGPU, massSpecies0GPU, massSpecies1GPU;
bool initializeFluidBinaryMixture();
bool freeBinaryMixture();
bool initializeBinaryMixtureGPU(); 
bool freeBinaryMixtureGPU();
bool createCellsBinaryMixture();
//Binary Minture Ends


//Binary Mixture Wall Begins
EXTERN_OTHER_FLUID_V bool setBinaryMixtureWall;
EXTERN_OTHER_FLUID_V double cWall0, cWall1; //Concentration in the wall
EXTERN_OTHER_FLUID_V double vxWall0, vxWall1;
EXTERN_OTHER_FLUID_V double vyWall0, vyWall1;
EXTERN_OTHER_FLUID_V double vzWall0, vzWall1;
EXTERN_OTHER_FLUID_V double densityWall0, densityWall1;
//Binary Mixture Wall Ends

//Giant Fluctuations Begins
EXTERN_OTHER_FLUID_V bool setGiantFluctuations;
EXTERN_OTHER_FLUID_V double soretCoefficient;
EXTERN_OTHER_FLUID_V double gradTemperature;
//Giant Fluctuations Ends

//Continuous Gradient Begins
EXTERN_OTHER_FLUID_V bool setContinuousGradient;
//Continuous Gradient Ends

//Incompressible Begins
EXTERN_OTHER_FLUID_V bool incompressible;
//Incompressible Ends

//Incompressible Boundary RK2 Begins
EXTERN_OTHER_FLUID_V bool incompressibleBoundaryRK2;
//Incompressible Boundary RK2 Ends

//quasiNeutrallyBuoyant Begins
EXTERN_OTHER_FLUID_V bool quasiNeutrallyBuoyant;
//quasiNeutrallyBuoyant Ends

//quasiNeutrallyBuoyant2D Begins
EXTERN_OTHER_FLUID_V bool quasiNeutrallyBuoyant2D;
//quasiNeutrallyBuoyant2D Ends

//quasiNeutrallyBuoyant4pt2D Begins
EXTERN_OTHER_FLUID_V bool quasiNeutrallyBuoyant4pt2D;
//quasiNeutrallyBuoyant4pt2D Ends

//IMEXRK Begins
EXTERN_OTHER_FLUID_V bool IMEXRK;
//IMEXRK Ends

//IncompressibleBinaryMixture Begins
EXTERN_OTHER_FLUID_V bool incompressibleBinaryMixture;
//IncompressibleBinaryMixture Ends

//IncompressibleBinaryMixtureMidPoint Begins
EXTERN_OTHER_FLUID_V bool incompressibleBinaryMixtureMidPoint;
//IncompressibleBinaryMixtureMidPoint Ends

//particlesWall Begins
EXTERN_OTHER_FLUID_V bool particlesWall;
//particlesWall Ends

//testJPS Begins
EXTERN_OTHER_FLUID_V int testJPS;
//testJPS Ends

//freeEnergyCompressibleParticles Begins
EXTERN_OTHER_FLUID_V int freeEnergyCompressibleParticles;
EXTERN_OTHER_FLUID_V double omega0;
//freeEnergyCompressibleParticles Ends

//semiImplicitCompressibleParticles Begins
EXTERN_OTHER_FLUID_V bool semiImplicitCompressibleParticles;
//semiImplicitCompressibleParticles Ends

//momentumCoupling Begins
EXTERN_OTHER_FLUID_V bool momentumCoupling;
//momentumCoupling Ends

//stokesLimit Begins
EXTERN_OTHER_FLUID_V bool stokesLimit;
EXTERN_OTHER_FLUID_V bool setExtraMobility;
EXTERN_OTHER_FLUID_V double extraMobility;
//stokesLimit Ends


