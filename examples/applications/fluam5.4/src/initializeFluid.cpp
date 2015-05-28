// Filename: initializeFluid.cpp
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
#include "parameters.h"
//#include "vectorial_calculus.h"
#include "temperature.h"



bool initializeFluid(){
  double v0, k, kx, ky, kz;
  double vx, vy, vz;

  vx = 0.;
  vy = 0.;
  vz = 0.;
  
  if(loadFluid==0){
    for(int i=0;i<ncells;i++){

      int fx = i % mx;
      int fy = (i % (my*mx)) / mx;
      int fz = i / (mx*my);

      cDensity[i] = densfluid;
      
      if(incompressibleBinaryMixture || incompressibleBinaryMixtureMidPoint)
      if(initfluid==5) { // Sharp discontinuity for mixing
        double width=1.0; // How wide/smooth to make the tanh profile
        // Profile for mixing with periodic BCs: a stripe in the middle
        c[i] = (1-concentration) + (2*concentration-1)/2* \
           ( tanh( (fy-double(my)/3+0.5) / width ) - tanh( (fy-2*double(my)/3+0.5) / width ));
        //cout << i << " c=" << c[i] << endl;
      }
      else if(initfluid==6) { // A smooth single-mode profile for mixing
        int mode=1;
	kx = 2*pi/double(mx) * mode;
	ky = 2*pi/double(my) * mode;
        c[i] = sin(kx*(fx+0.5)) * sin(ky*(fy+0.5));
        //c[i] = sin(kx*(fx+0.5)) ;
      }
      else if(initfluid==7) { // Several lines next to each other
        double width=1.0; // How wide/smooth to make the tanh profile
        // Profile for mixing with periodic BCs: a stripe in the middle
        c[i] = (1-concentration) + (2*concentration-1)/2* ( \
           ( tanh( (fy-double(my)/2+6*width+12.5) / width ) - tanh( (fy-double(my)/2-6*width+12.5) / width )) \
          -( tanh( (fy-double(my)/2+6*width-11.5) / width ) - tanh( (fy-double(my)/2-6*width-11.5) / width )) \
           );
        //cout << i << " c=" << c[i] << endl;
      }
      else if(initfluid==1) { // Equilibrium concentration fluctuations
         double cvar = concentration * (1-concentration) * (massSpecies1*(1-concentration) + massSpecies0*concentration);
         c[i] = concentration + sqrt(cvar/cVolume)*gauss();
         
      }
      else {
	c[i] = concentration;
      }
          

      switch(initfluid){
      case 0:
	//******************************************
	//Fluido con velocidad cero
	//******************************************
	cvx[i] = vx0;
	cvy[i] = vy0;
	cvz[i] = vz0;
	break;
      case 1:
      case 5:
      case 6:
      case 7:
	//!*************************************************
	//! Fluido en equilibrio a T != 0
	//!*************************************************
	cvx[i] = sqrt(temperature/(cDensity[i]*cVolume))*gauss();
	cvy[i] = sqrt(temperature/(cDensity[i]*cVolume))*gauss();
	cvz[i] = sqrt(temperature/(cDensity[i]*cVolume))*gauss();
	vx += cvx[i];
	vy += cvy[i];
	vz += cvz[i];
	break;
      case 2:
	//!*************************************************
	//! Onda transversal y temperatura 0
	//! vx = v0 sin(kz)exp(-nuk**2t)
	//!*************************************************
	v0 = 0.04;
	k = 2*pi/lz * 1;
	cvx[i] = vx0;
	cvy[i] = vy0 + v0*sin(k*crz[i]);//0.;
	cvz[i] = vz0;//0.;
	break;
      case 3:
	//!*************************************************
	//! Onda longitudinal y temperatura 0
	//! vz = v0 sin(kz)cos(c_t k t)exp(-gamma_t k**2 t
	//! c_t = velocidad del sonido
	//! gamma_t = absorción isoterma del sonido
	//!*************************************************
	v0 = 0.004;
	k = 2*pi/lz * 8;
	cvx[i] = 0.;
	cvy[i] = 0.;//0.;
	cvz[i] = v0 * sin(k*(crz[i]+0.5*lz/double(mz)));
	break;
      case 4:
	//c[0].density = 1.2;
	//c[0].mass = c[0].density * c[0].volume;
	cvx[i] = 0.425;
	cvy[i] = 0.;
	cvz[i] = 0.;
	/*c[555].v[0] = 2.;
	  c[555].v[1] = 2.;
	  c[555].v[2] = 2.;
	  c[0] .v[0] = -2.;
	  c[0] .v[1] = -2.;
	  c[0] .v[2] = -2.;*/
	//c[0].v[0] = 10.;
	break;
      default:
	for(int i=0;i<ncells;i++){
	  cvx[i] = 0;
	  cvy[i] = 0;
	  cvz[i] = 0;
	}
	break;
      }
      //c[i].p[0] = c[i].v[0] * c[i].mass;
      //c[i].p[1] = c[i].v[1] * c[i].mass;
      //c[i].p[2] = c[i].v[2] * c[i].mass;
    }
    
    if(initfluid == 1 || initfluid>=5) {
      vx = vx/double(ncells);
      vy = vy/double(ncells);
      vz = vz/double(ncells);
      for(int i=0;i<ncells;i++){
	cvx[i] = cvx[i] - vx + vx0;
	cvy[i] = cvy[i] - vy + vy0;
	cvz[i] = cvz[i] - vz + vz0;
        //cout << i << " " << cvx[i] << " " << cvy[i] << " " << cvz[i] << endl;
      }
    }
  }
  else {
    ifstream fileinput(loadFluidFile.c_str());
    string word;
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    fileinput >> word >> densfluid;
    fileinput >> word >> word >> pressurea0 >> pressurea1 >> pressurea2;
    //getline(fileinput,word);//////
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    for(int i=0;i<ncells;i++) {
      fileinput >> cDensity[i];
      if(incompressibleBinaryMixture)
	fileinput >> c[i];
      fileinput >> cvx[i]  >> cvy[i] >> cvz[i];
    }
    fileinput.close();   
  }
  
  
  return 1;
}
