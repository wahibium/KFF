// Filename: createParticles.cpp
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


//#define GLOBALS_PARTICLES 1
#include <cstring>
#include <math.h>
#include "header.h"
#include "particles.h"
#include "fluid.h"
#include "parameters.h"
#include "cells.h"

const string wnothing="#";

const string wsigma="sigma";
const string wrho="rho";
const string wsigmap="sigmap";



bool createParticles(){

  //Create particles in a simple cubic lattice
  if(loadparticles==0){
    rxParticle = new double [np];
    ryParticle = new double [np];
    rzParticle = new double [np];
    vxParticle = new double [np];
    vyParticle = new double [np];
    vzParticle = new double [np];
    //This is for the interpolate velocity
    vxParticleI = new double [np];
    vyParticleI = new double [np];
    vzParticleI = new double [np];
    simpleCubic();
    double vx, vy, vz;
    vx = 0;
    vy = 0;
    vz = 0;
    for(int i=0;i<np;i++){
      vxParticle[i] = 0.;//sqrt(temperature/mass) * gauss();
      vyParticle[i] = 0.;//sqrt(temperature/mass) * gauss();
      vzParticle[i] = 0.;//sqrt(temperature/mass) * gauss();
      vx = vx + vxParticle[i];
      vy = vy + vyParticle[i];
      vz = vz + vzParticle[i];
    }
    vx = vx/double(np);
    vy = vy/double(np);
    vz = vz/double(np);	
    
    for(int i=0;i<np;i++){
      vxParticle[i] -= vx;
      vyParticle[i] -= vy;
      vzParticle[i] -= vz;
    }
  }
  else if(loadparticles==1){
    string nullString ="";
    if(particlescoor==nullString){
      cout << "ERROR NO PARTICLESCOOR FILE FOUND" << endl;
      return 0;
    }
    ifstream filecoor(particlescoor.c_str());
    filecoor >> np;
    rxParticle = new double [np];
    ryParticle = new double [np];
    rzParticle = new double [np];
    vxParticle = new double [np];
    vyParticle = new double [np];
    vzParticle = new double [np];
    //This is for the interpolate velocity
    vxParticleI = new double [np];
    vyParticleI = new double [np];
    vzParticleI = new double [np];
    for(int i=0;i<np;i++)
      filecoor >> rxParticle[i] >> ryParticle[i] >> rzParticle[i];
    filecoor.close();
    if(particlesvel==nullString){
      for(int i=0;i<np;i++){
	vxParticle[i] = 0.;
	vzParticle[i] = 0.;
	vyParticle[i] = 0.;
      }
    }
    else{
      filecoor.open(particlesvel.c_str());
      int npaux;
      filecoor >> npaux;
      if(npaux!=np){
	cout << "ERROR NP DIFFERENT IN PARTICLESCOOR AND PARTICLESVEL" << endl;
	return 0;
      }
      for(int i=0;i<np;i++)
	filecoor >> vxParticle[i] >> vyParticle[i] >> vzParticle[i];
      //cout << "ERROR, LOAD PARTICLES IS NOT YET WRITED" << endl;
      //return;
    }
    filecoor.close();
  }  
  volumeParticle = volumeParticle/cVolume;




  cout << "CREATE PARTICLES :              DONE " << endl;


  return 1;
}
