// Filename: saveFluidFinalConfiguration.cu
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
#include "headerOtherFluidVariables.h"
#include "fluid.h"
#include "cells.h"
#include "particles.h"
//#include "visit_writer.h"
//#include "visit_writer.c"
#include "hydroAnalysis.h"

bool saveFluidFinalConfiguration(){
  ofstream file;
  
  string savefile;
  savefile = outputname + ".fluidFinalConfiguration";
  file.open(savefile.c_str());
  file.precision(15);
  file << "Number of cells " << ncells << endl;
  file << "mx my mz" << endl;
  file << mx << " " << my << " " << mz << endl;
  file << "lx ly lz" << endl;
  file << lx << " " << ly << " " << lz << endl;
  file << "Density " << densfluid << endl;
  file << "Pressure parameters " << pressurea0 << " " << pressurea1 << " " << pressurea2 << endl;
  file << "Thermostat " << thermostat << endl;
  file << "Temperature " << temperature << endl;
  file << "Cells properties " << endl;
  file << "Density " << endl;
  file << "Velocity" << endl;


  int i;
  for(int j=0;j<ncells;j++){
    if(particlesWall){
      //go to index tanking into account ghost cells
      int fx = j % mx;
      int fy = (j % (my*mx)) / mx;
      int fz = j / (mx*my);
      fy++;
      i = fx + fy*mxt + fz*mxt*myt;
    }
    else{
      i = j;
    }
    file << cDensity[i] << endl;
    if(incompressibleBinaryMixture || incompressibleBinaryMixtureMidPoint)
      file << c[i] << endl;
    file << cvx[i] << " " << cvy[i] << " " << cvz[i] << endl;
  }
    
  file.close();


  return 1;

}
