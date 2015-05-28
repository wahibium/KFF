// Filename: schemeCompressibleParticles.cu
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


#include "header.h"
#include "cells.h"
#include "particles.h"
#include "boundary.h"


bool schemeCompressibleParticles(){
  
  //Create fluid cells
  if(!createCells()) return 0;

  //Create boundaries
  if(setboundary)
    if(!createBoundaries()) return 0;

  //Create particles
  if(setparticles)
    if(!createParticles()) return 0;

  //New bonded forces
  if(bondedForces)
    if(!initializeBondedForces()) return 0;

  //Initialize the fluid
  if(!initializeFluid()) return 0;
  
  //Create fluid cells in the GPU
  if(!createCellsGPU()) return 0;
  
  //Initialize fluid in the GPU
  if(!initializeFluidGPU()) return 0;

  //Create boundaries GPU
  if(!createBoundariesGPU()) return 0;

  //New bonded forces
  if(bondedForces)
    if(!createBondedForcesGPU()) return 0;

  //Initialize save functions
  if(!saveFunctionsSchemeBoundary(0,0)) return 0;

  //Run the simulation
  if(!runSchemeCompressibleParticles()) return 0;

  //Close save functions
  if(!saveFunctionsSchemeBoundary(2,0)) return 0;

  //New bonded forces
  if(bondedForces)
    if(!freeBondedForcesGPU()) return 0;

  //NEW bonded forces
  if(bondedForces)
    if(!freeBondedForces()) return 0;
 
  //Free Memory GPU
  if(!freeCellsGPU()) return 0;
  
  //Free boundaries GPU
  if(!freeBoundariesGPU()) return 0;

  //Free boundaries
  if(setboundary)
    if(!freeBoundaries()) return 0;

  //Free particles
  if(setparticles)
    if(!freeParticles()) return 0;
 
  //Free memory
  if(!freeMemoryCompressibleParticles()) return 0;
  

  return 1;
}






bool freeMemoryCompressibleParticles(){

  delete[] cvx;
  delete[] cvy;
  delete[] cvz;
  delete[] crx;
  delete[] cry;
  delete[] crz;
  delete[] cDensity;

  

  cout << "FREE MEMORY :                   DONE" << endl;


  return 1;
}


