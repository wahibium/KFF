// Filename: createBoundaries.cpp
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



//#define GLOBALS_BOUNDARY 1

#include "header.h"
#include "boundary.h"


bool createBoundaries(){

  //LOAD ARCHIVE
  ifstream fileinput;
  fileinput.open(fileboundary.c_str());
  fileinput >> nboundary;
  if(nboundary==0) return 0;
  rxboundary = new double [nboundary];
  ryboundary = new double [nboundary];
  rzboundary = new double [nboundary];
  vxboundary = new double [nboundary];
  vyboundary = new double [nboundary];
  vzboundary = new double [nboundary];
  for(int i=0;i<nboundary;i++)
    fileinput >> rxboundary[i] >> ryboundary[i] >> rzboundary[i] >> vxboundary[i] >> vyboundary[i] >> vzboundary[i];
  fileinput.close();
  

  cout << "CREATE BOUNDARY :               DONE" << endl;

  return 1;
}

bool freeBoundaries(){
  delete[] rxboundary;
  delete[] ryboundary;
  delete[] rzboundary;
  delete[] vxboundary;
  delete[] vyboundary;
  delete[] vzboundary;
  cout << "FREE BOUNDARY :                 DONE" << endl;
}
