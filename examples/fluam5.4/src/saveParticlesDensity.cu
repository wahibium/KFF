// Filename: saveParticlesDensity.cu
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
#include "particles.h"
#include "headerOtherFluidVariables.h"


static ofstream fileParticlesDensity;
bool saveParticlesDensity(int index, long long step){
  fileParticlesDensity.precision(15);

  if(index==0){
    string savefile;
    savefile = outputname + ".densityParticle";
    fileParticlesDensity.open(savefile.c_str());
    fileParticlesDensity << "#NUMBER PARTICLES " << np << endl;
  }
  else if(index==1){
    fileParticlesDensity << step * dt << endl;
    for(int i=0;i<np;i++){
      fileParticlesDensity << vxParticleI[i] << endl;
    }
  }
  else if(index==2){
    fileParticlesDensity.close();
  }
  return 1;
}

