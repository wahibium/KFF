// Filename: schemeIncompressible.cu
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


bool schemeIncompressible(){
  //Create fluid cells
  if(!createCells()) return 0;

  //Initialize the fluid
  if(!initializeFluid()) return 0;

  //Create fluid cells in the GPU
  if(!createCellsIncompressibleGPU()) return 0;

  //Initialize fluid in the GPU
  if(!initializeFluidIncompressibleGPU()) return 0;

  //Initialize save functions
  if(!saveFunctionsSchemeIncompressible(0)) return 0;
  
  //Run the simulation
  if(!runSchemeIncompressible()) return 0;
      
  //Close save functions
  if(!saveFunctionsSchemeIncompressible(2)) return 0;

  //Free Memory GPU
  if(!freeCellsIncompressibleGPU()) return 0;

  //Free memory
  if(!freeMemoryIncompressible()) return 0;
  

  return 1;
}



bool freeMemoryIncompressible(){

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
