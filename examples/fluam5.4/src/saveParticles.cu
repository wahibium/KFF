// Filename: saveParticles.cu
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


static ofstream file;
static ofstream fileVelocity;
//This is for the interpolate velocity
static ofstream fileVelocityI;


bool saveParticles(int option, long long step){
  fileVelocity.precision(15);  
  file.precision(15);  
  //This is for the interpolate velocity
  fileVelocityI.precision(15);
  if(option == 0){
    string savefile;
    savefile = outputname +  ".particles";
    file.open(savefile.c_str());
    file << "#NUMBER PARTICLES " << np << endl;

    savefile = outputname + ".velocityParticles";
    fileVelocity.open(savefile.c_str());
    fileVelocity << "#NUMBER PARTICLES " << np << endl;
    
    //This is for the interpolate velocity
    if(quasiNeutrallyBuoyant){
      savefile = outputname + ".velocityParticlesI";
      fileVelocityI.open(savefile.c_str());
      fileVelocityI << "#NUMBER PARTICLES " << np << endl;
    }
  }
  else if(option == 1){
    file << step * dt << endl;
    fileVelocity << step * dt << endl;
    //This is for the interpolate velocity
    if(quasiNeutrallyBuoyant) 
      fileVelocityI << step * dt << endl;
    for(int i=0;i<np;i++){
      file << rxParticle[i] << " " << ryParticle[i] << " " << rzParticle[i] << endl;
      fileVelocity << vxParticle[i] << " " << vyParticle[i] << " " << vzParticle[i] << endl;
      //This is for the interpolate velocit
      if(quasiNeutrallyBuoyant || quasiNeutrallyBuoyant2D) 
	fileVelocityI << vxParticleI[i] << " " << vyParticleI[i] << " " << vzParticleI[i] << endl;
    }
  }
  else if(option == 2){
    file.close();
    fileVelocity.close();
    //This is for the interpolate velocity
    if(quasiNeutrallyBuoyant) 
      fileVelocityI.close();
  }
    
  return 1;
}
