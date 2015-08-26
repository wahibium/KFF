// Filename: schemeRK3Ghost.cu
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


bool schemeRK3Ghost(){
  
  //Create fluid cells
  if(!createCellsGhost()) return 0;

  //Initialize the fluid
  if(!initializeFluidGhost()) return 0;
  
  //Create fluid cells in the GPU
  if(!createCellsGhostGPU()) return 0;
  
  //Initialize fluid in the GPU
  if(!initializeFluidGhostGPU()) return 0;

  //Initialize save functions
  if(!saveFunctionsSchemeRK3Ghost(0)) return 0;

  //Run the simulation
  if(!runSchemeRK3Ghost()) return 0;

  //Close save functions
  if(!saveFunctionsSchemeRK3Ghost(2)) return 0;

  //Free Memory GPU
  if(!freeCellsGhostGPU()) return 0;
  
  //Free memory
  if(!freeMemoryGhost()) return 0;
  

  return 1;
}






bool freeMemoryGhost(){

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


