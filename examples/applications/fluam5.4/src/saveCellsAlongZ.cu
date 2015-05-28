// Filename: header.h
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
#include "headerOtherFluidVariables.h"


static ofstream fileCellsAlongZ;


bool saveCellsAlongZ(int index){


  fileCellsAlongZ.precision(15);  

  if(index == 0){
    string savefile;
    savefile = outputname +  ".cellsZ";
    fileCellsAlongZ.open(savefile.c_str());
    fileCellsAlongZ << "#NUMBER CELLS Z   " << mz << endl;
  }
  else if(index == 1){
    
    double density, vx, vy, vz;
    int cell;
    
    fileCellsAlongZ << step * dt << endl;
    for(int i=0;i<mz;i++){
      density=0;
      vx=0;
      vy=0;
      vz=0;
      for(int j=0;j<my;j++){
	for(int k=0;k<mx;k++){
	  cell = k + j*mx + i*mx*my;
	  density += cDensity[cell];
	  vx += cvx[cell];
	  vy += cvy[cell];
	  vz += cvz[cell];
	}
      }
      density /= mx*my;
      vx /= mx*my;
      vy /= mx*my;
      vz /= mx*my;
      fileCellsAlongZ << crz[cell] << "  " << density << "  " << vx << "  " << vy << "  " << vz << endl;
    }
  }
  else if(index == 2){
    fileCellsAlongZ.close();
  }
    
  return 1;



}




