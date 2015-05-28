// Filename: header.h
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


#include <iostream>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>

using namespace std;

#ifdef GLOBAL
#define EXTERN_GLOBAL
#else
#define EXTERN_GLOBAL extern
#endif

//NEW_PARAMETER
EXTERN_GLOBAL double identity_prefactor;
EXTERN_GLOBAL int setDevice;
EXTERN_GLOBAL bool setparticles;
EXTERN_GLOBAL bool thermostat;
EXTERN_GLOBAL double dt;
EXTERN_GLOBAL long long numsteps;
EXTERN_GLOBAL long long numstepsRelaxation;
EXTERN_GLOBAL int samplefreq, savefreq;
EXTERN_GLOBAL string outputname;
EXTERN_GLOBAL bool savedensity;
EXTERN_GLOBAL int nsavedensity;
EXTERN_GLOBAL int* cellsavedensity;
EXTERN_GLOBAL double volumeboundaryconst;
EXTERN_GLOBAL int maxNumberPartInCellNonBonded, maxNumberPartInCell;
EXTERN_GLOBAL int waveSourceIndex;
EXTERN_GLOBAL string particlescoor;
EXTERN_GLOBAL string particlesvel;
EXTERN_GLOBAL string loadFluidFile;
EXTERN_GLOBAL int loadFluid;
EXTERN_GLOBAL int nCheck;
EXTERN_GLOBAL bool setCheckVelocity;
EXTERN_GLOBAL bool setSaveFluid;
EXTERN_GLOBAL int seed;
EXTERN_GLOBAL bool bool_seed;
EXTERN_GLOBAL long long step;
EXTERN_GLOBAL bool setGhost;
EXTERN_GLOBAL bool setboundary;
EXTERN_GLOBAL string fileCheckVelocity;
EXTERN_GLOBAL bool setSaveVTK;
EXTERN_GLOBAL bool computeNonBondedForces;


bool loadDataMain(int argc, char* argv[]);
bool writeDataMain();
bool initializeRandomNumber();
bool createBoundaries();
bool createParticles();
//NEW bonded forces
bool initializeBondedForces();
bool freeBondedForces();
bool createBondedForcesGPU();
bool freeBondedForcesGPU();

bool initializeRandomGPU();
bool createBoundariesGPU();
bool freeBoundaries();
bool createParticlesGPU();
bool run();
bool freeMemoryGPU();
bool freeMemory();
bool freeMemoryBoundary();
bool freeMemoryThermostat();
bool initializeOtherFluidVariables();
bool freeOtherFluidVariables();
bool initializeOtherFluidVariablesGPU();
bool freeOtherFluidVariablesGPU();
bool configurationSaveFunctions(int i);
bool schemeThermostat();
bool saveTime(int index);
bool saveSeed();
bool schemeBinaryMixture();
bool hydroAnalysis(int counter);
bool hydroAnalysisGhost(int counter);
bool hydroAnalysisIncompressible(int counter);
bool schemeBinaryMixtureWall();
bool cudaDevice();
bool saveFluidFinalConfiguration();
bool saveFluidVTK(int option);

//Functions for pseudorandom numbers
void RANTEST(int Iseed);
double RANFRK();
double XRANDXXX();
void RANSET(int Iseed);
void RSTART(int Iseeda);
double RCARRY();
double gauss();

bool init_random_gpu(int SEED);
bool free_random_gpu();

int modu(int x, int y);
void simpleCubic();


//schemeRK3Ghost
bool initializeFluidGhost();
bool createCellsGhost();
bool schemeRK3Ghost();
bool freeMemoryRK3GPU();
bool runSchemeRK3Ghost();
bool createCellsGhostGPU();
bool initializeFluidGhostGPU();
bool freeCellsGhostGPU();
bool initializeFluidGhostGPU();
//bool temperatureGhost(int index);
bool saveFunctionsSchemeRK3Ghost(int index);
bool temperatureGhost(int index);
bool freeMemoryGhost();

//schemeRK3
bool initializeFluid();
bool createCells();
bool schemeRK3();
bool freeMemoryRK3GPU();
bool runSchemeRK3();
bool createCellsGPU();
bool initializeFluidGPU();
bool freeCellsGPU();
bool initializeFluidGhostGPU();
//bool temperatureGhost(int index);
bool saveFunctionsSchemeRK3(int index);
bool temperatureFunction(int index);

//schemeThermostat
bool saveFunctionsSchemeThermostat(int index);
bool runSchemeThermostat();
bool saveCellsAlongZ(int index);

//schemeBoundary
bool schemeBoundary();
bool freeParticles();
bool createBoundariesGPU();
bool freeBoundariesGPU();
bool temperatureBoundary(int index);
bool saveParticles(int index, long long step);
bool saveFunctionsSchemeBoundary(int index, long long step);
bool initForcesNonBonded();
bool freeDelta();
bool runSchemeBoundary();
//bool covarianceKernel(int index);

//schemeBinaryMixture
bool createCellsBinaryMixtureGPU();
bool freeCellsBinaryMixtureGPU();
bool initializeFluidBinaryMixtureGPU();
bool saveFunctionsSchemeBinaryMixture(int index);
bool runSchemeBinaryMixture();
bool totalConcentration(int index);

//schemeBinaryMixtureWall
bool runSchemeBinaryMixtureWall();
bool copyToGPUBinaryMixtureWall();
bool initializeFluidBinaryMixtureWall();

//SchemeGiantFluctuations
bool initializeFluidGiantFluctuations();
bool schemeGiantFluctuations();
bool runSchemeGiantFluctuations();
bool copyToGPUGiantFluctuations();

//SchemeContinuousGradient
bool schemeContinuousGradient();
bool runSchemeContinuousGradient();

//SchemeIncompressible
bool schemeIncompressible();
bool freeMemoryIncompressible();
bool runSchemeIncompressible();
bool saveFunctionsSchemeIncompressible(int index);
     
//SchemeIncompressibleBoundary
bool schemeIncompressibleBoundary();
bool createCellsIncompressibleGPU();
bool initializeFluidIncompressibleGPU();
bool freeMemoryIncompressibleBoundary();
bool freeCellsIncompressibleGPU();
bool runSchemeIncompressibleBoundary();
bool saveFunctionsSchemeIncompressibleBoundary(int index, long long step);

//SchemeIncompressibleBoundaryRK2
bool schemeIncompressibleBoundaryRK2();
bool createBoundariesRK2GPU();
bool freeBoundariesRK2GPU();
bool freeMemoryIncompressibleBoundaryRK2();
bool runSchemeIncompressibleBoundaryRK2();

//SchemeQuasiNeutrallyBuoyant
bool schemeQuasiNeutrallyBuoyant();
bool freeMemoryQuasiNeutrallyBuoyant();
bool runSchemeQuasiNeutrallyBuoyant();

//SchemeQuasiNeutrallyBuoyant2D
bool schemeQuasiNeutrallyBuoyant2D();
bool freeMemoryQuasiNeutrallyBuoyant2D();
bool runSchemeQuasiNeutrallyBuoyant2D();
bool saveFunctionsSchemeIncompressibleBoundary2D(int index, long long step);
bool temperatureBoundary2D(int index);

//SchemeQuasiNeutrallyBuoyant4pt2D
bool createBoundaries4ptGPU();
bool schemeQuasiNeutrallyBuoyant4pt2D();
bool freeMemoryQuasiNeutrallyBuoyant4pt2D();
bool runSchemeQuasiNeutrallyBuoyant4pt2D();


//schemeCompressibleParticles
bool schemeCompressibleParticles();
bool freeMemoryCompressibleParticles();
bool runSchemeCompressibleParticles();

//SchemeIncompressibleBinaryMixture
bool createCellsIncompressibleBinaryMixtureGPU();
bool schemeIncompressibleBinaryMixture();
bool freeMemoryIncompressibleBinaryMixture();
bool freeCellsIncompressibleBinaryMixtureGPU();
bool createCellsIncompressibleBinaryMixtureGPU();
bool runSchemeIncompressibleBinaryMixture();
bool hydroAnalysisIncompressibleBinaryMixture(int counter);
bool saveFunctionsSchemeIncompressibleBinaryMixture(int index);

//SchemeIncompressibleBinaryMixtureMidPoint
bool schemeIncompressibleBinaryMixtureMidPoint();
bool runSchemeIncompressibleBinaryMixtureMidPoint();
bool freeMemoryIncompressibleBinaryMixtureMidPoint();

//SchemeParticlesWall
bool schemeParticlesWall();
bool freeMemoryParticlesWall();
bool initializeFluidParticlesWall();
bool saveFunctionsSchemeParticlesWall(int index, long long step);
bool temperatureParticlesWall(int index);
bool hydroAnalysisParticlesWall(int counter);
bool runSchemeParticlesWall();

//////
bool temperatureParticlesWall2(int index);
     
//SchemeTestJPS
bool schemeTestJPS();
bool freeMemoryTestJPS();
bool saveFunctionsSchemeTestJPS(int index, long long step);
bool runSchemeTestJPS();

//SchemeFreeEnergyCompressibleParticles
bool schemeFreeEnergyCompressibleParticles();
bool freeMemoryFreeEnergyCompressibleParticles();
bool runSchemeFreeEnergyCompressibleParticles();
bool saveParticlesDensity(int index, long long step);

//semiImplicitCompressibleParticles
bool schemeSemiImplicitCompressibleParticles();
bool freeMemorySemiImplicitCompressibleParticles();
bool createCellsSemiImplicitCompressibleParticlesGPU();
bool freeCellsSemiImplicitCompressibleParticlesGPU();
bool runSchemeSemiImplicitCompressibleParticles();
bool saveFunctionsSchemeSemiImplicitCompressibleParticles(int index, long long step);

//momentumCoupling
bool freeMemoryMomentumCoupling();
bool schemeMomentumCoupling();
bool runSchemeMomentumCoupling();

//stokesLimit
bool schemeStokesLimit();
bool saveFunctionsSchemeStokesLimit(int index, long long step);
bool runSchemeStokesLimit();
bool createCellsStokesLimitGPU();
bool freeCellsStokesLimitGPU();



