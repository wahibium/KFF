// Filename: totalConcentration.cpp
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


bool totalConcentration(int index){
  double density1=0;
  for(int j=0;j<ncells;j++){
    int i, fx, fy, fz;
    fx = j % mx;
    fy = (j % (mx*my)) / mx;
    fz = j / (mx*my);
    //go to index tanking into account ghost cells
    fx++;
    fy++;
    fz++;
    i = fx + fy*mxt + fz*mxt*myt;
    //cout << "KKK " << i << " " << cDensity[i] << " " << density1 << endl;
    density1 += cVolume * cDensity[i] * c[i];
  }

  cout.precision(15);
  cout << "DENSITY SPECIES 1  " << density1 << endl;
  cout.precision(6);

  if(index==2){
    string Nombre;
    Nombre =  outputname + ".densitySpecies1";
    ofstream fileSave(Nombre.c_str());
    fileSave.precision(15);
    fileSave << "DENSITY SPECIES 1  " << density1 << endl;
    fileSave.close();
  }

  return 1;
}
