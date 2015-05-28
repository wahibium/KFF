// Filename: schemeBinaryMixtureWall.cu
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


//Scheme for the binary mixture using 
//RK3 with 2RNG and ghost cells

#include "header.h"
#include "cells.h"
#include "headerOtherFluidVariables.h"

bool freeMemoryBinaryMixtureWall();

bool schemeBinaryMixtureWall(){
  
  //Create fluid cells
  if(!createCellsBinaryMixture()) return 0;

  //Initialize the fluid
  if(!initializeFluidBinaryMixtureWall()) return 0;
  
  //Initialize random number generator in the GPU
  //if(!init_random_gpu(seed)) return 0;

  //Create fluid cells in the GPU
  if(!createCellsBinaryMixtureGPU()) return 0;

  //Copy Aditional variables to GPU
  if(!copyToGPUBinaryMixtureWall()) return 0;
  
  //Initialize fluid in the GPU
  if(!initializeFluidBinaryMixtureGPU()) return 0;

  //Initialize save functions
  if(!saveFunctionsSchemeBinaryMixture(0)) return 0;

  //Run the simulation
  if(!runSchemeBinaryMixtureWall()) return 0;

  //Close save functions
  if(!saveFunctionsSchemeBinaryMixture(2)) return 0;

  //Free Memory GPU
  if(!freeCellsBinaryMixtureGPU()) return 0;

  //Free memory rnadom GPU
  //if(!free_random_gpu()) return 0;
  
  //Free memory
  if(!freeMemoryBinaryMixtureWall()) return 0;
  

  return 1;
}






bool freeMemoryBinaryMixtureWall(){

  delete[] cvx;
  delete[] cvy;
  delete[] cvz;
  delete[] crx;
  delete[] cry;
  delete[] crz;
  delete[] cDensity;
  delete[] c;
  

  cout << "FREE MEMORY :                   DONE" << endl;


  return 1;
}


