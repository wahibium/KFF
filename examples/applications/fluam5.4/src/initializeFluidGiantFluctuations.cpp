// Filename: initializeFluidGiantFluctuations.cpp
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
#include "fluid.h"
#include "cells.h"
#include "parameters.h"
//#include "vectorial_calculus.h"
#include "temperature.h"
#include "headerOtherFluidVariables.h"


bool initializeFluidGiantFluctuations(){
  double v0, k;
  double vx, vy, vz;
  int fx, fy, fz;
  double dc;
  vx = 0.;
  vy = 0.;
  vz = 0.;
  
  dc = soretCoefficient * gradTemperature * ly;

  if((massSpecies0==0)||(massSpecies1==0)||(diffusion==0)){
    cout << "(massSpecies0==0)||(massSpecies1==0)||(diffusion==0)" << endl;
    return 0;
  }

  if(loadFluid==0){
    for(int j=0;j<ncells;j++){
      int i;
      fx = j % mx;
      fy = (j % (my*mx)) / mx;
      fz = j / (mx*my);
      //go to index tanking into account ghost cells
      fx++;
      fy++;
      fz++;
      i = fx + fy*mxt + fz*mxt*myt;

      cDensity[i] = densfluid;
      if(gradTemperature==0) {
	c[i] = concentration;
      }
      else{
	c[i] = concentration * dc * exp(-dc*(fy-0.5)/double(my))/(1-exp(-dc));
	//The next line fixed the init bias for
	//<c>=0.018, h=0.1, mx=128, my=32, gradT=174.
	c[i] += -4.4797e-7 +
	  1.1985e-6 * ((fy-0.5)*0.1/double(my)) +
	  6.8284e-4 * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) -
	  1.6661e-2 * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) +
	  0.16417 * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) -
	  0.6038 * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) * ((fy-0.5)*0.1/double(my)) *
	  ((fy-0.5)*0.1/double(my));
	
	//c[i] += -4.918e-7 + 1.2248e-5*((fy-0.5)/double(my)*0.1)
	//  + 7.8826e-6 * ((fy-0.5)/double(my)*0.1)*((fy-0.5)/double(my)*0.1) 
	//  - 5.8778e-4 * ((fy-0.5)/double(my)*0.1)*((fy-0.5)/double(my)*0.1)*((fy-0.5)/double(my)*0.1);
	//c[i] = concentration;
      }
      //cMass[i] = cDensity[i] * cVolume;
      //c[i].pressure = pressure(c[i].density);
      
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
	//!*************************************************
	//! Fluido en equilibrio a T != 0
	//!*************************************************
	cvx[i] = sqrt(temperature/(cDensity[i]*cVolume))*gauss();
	cvy[i] = sqrt(temperature/(cDensity[i]*cVolume))*gauss();
	cvz[i] = sqrt(temperature/(cDensity[i]*cVolume))*gauss();
	if(fy!=(myt-2)){
	  vx += cvx[i];
	  vy += cvy[i];
	  vz += cvz[i];
	}
	else{
	  vx += cvx[i];
	  vz += cvz[i];
	}
	break;
      case 2:
	//!*************************************************
	//! Onda transversal y temperatura 0
	//! vx = v0 sin(kz)exp(-nuk**2t)
	//!*************************************************
	v0 = 2.04;
	k = 2*pi/lz;
	cvx[i] = v0*sin(k*crz[i]);
	cvy[i] = 0.;
	cvz[i] = 0.;
	break;
      case 3:
	//!*************************************************
	//! Onda longitudinal y temperatura 0
	//! vz = v0 sin(kz)cos(c_t k t)exp(-gamma_t k**2 t
	//! c_t = velocidad del sonido
	//! gamma_t = absorción isoterma del sonido
	//!*************************************************
	v0 = 0.0204;
	k = 2*pi/lz;
	cvx[i] = 0.;
	cvy[i] = 0.;
	cvz[i] = v0 * sin(k*crz[i]);
	break;
      case 4:
	//c[0].density = 1.2;
	//c[0].mass = c[0].density * c[0].volume;
	cvx[i] = 0.00000001;
	cvy[i] = 0.;
	cvz[i] = 0.;
	//vx += cvx[i];
	//vy += cvy[i];
	//vz += cvz[i];

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
    
    if(initfluid == 1){
      vx = vx/double(ncells);
      vy = vy/double((ncells-mx*mz));
      vz = vz/double(ncells);
      
      for(int j=0;j<ncells;j++){
	int i;
	fx = j % mx;
	fy = (j % (my*my)) / mx;
	fz = j / (mx*my);
	//go to index tanking into account ghost cells
	fx++;
	fy++;
	fz++;
	i = fx + fy*mxt + fz*mxt*myt;	
	if(fy!=(myt-2)){
	  cvx[i] = cvx[i] - vx + vx0;
	  cvy[i] = cvy[i] - vy + vy0;
	  cvz[i] = cvz[i] - vz + vz0;
	}
	else{
	  cvx[i] = cvx[i] - vx + vx0;
	  cvz[i] = cvz[i] - vz + vz0;
	}
      }
      vx=0;vy=0;vz=0;
      for(int j=0;j<ncells;j++){
	int i;
	fx = j % mx;
	fy = (j % (my*my)) / mx;
	fz = j / (mx*my);
	//go to index tanking into account ghost cells
	fx++;
	fy++;
	fz++;
	i = fx + fy*mxt + fz*mxt*myt;	
	if(fy!=(myt-2)){
	  vx += cvx[i];
	  vy += cvy[i];
	  vz += cvz[i];
	}
	else{
	  vx += cvx[i];
	  vz += cvz[i];
	}
      }
      cout << "KKKK " << vx << " " << vy << " " << vz << endl;
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
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    getline(fileinput,word);
    cout << word << endl;
    for(int i=0;i<ncells;i++) {
      fileinput >> cDensity[i];
      fileinput >> cvx[i]  >> cvy[i] >> cvz[i];
      //c[i].mass = c[i].density * c[i].volume;
      //cout << i << " " << c[i].density << endl;
    }
    fileinput.close();   
  }
  
  
  return 1;
}
