// Filename: temperatureBoundary.cu
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


//!***********************************
//! Temperature
//!***********************************
#include <math.h> 
#include "header.h"
#include "fluid.h"
#include "temperature.h"
#include "cells.h"
#include "particles.h"

bool temperatureBoundary2D(int index){

  int n;
  double px, py;
  double pxF, pyF;
  double pxP, pyP;
  double massTotal; 

  double t, vmedx, vmedy;
  double tx, ty;
  
  if(index==0){
    return 1;
  }
  else if(index==1){
    n = step / samplefreq;
    
    vmedx = 0.;
    vmedy = 0.;
    
    massTotal = 0;
    px=0; py=0; 
    pxF=0; pyF=0;
    pxP=0; pyP=0;
    
    
    for(int i=0;i<ncells;i++){
      vmedx = vmedx + cvx[i]; 
      vmedy = vmedy + cvy[i];
      massTotal += cVolume * cDensity[i];
    }
    
    vmedx = vmedx/double(ncells);
    vmedy = vmedy/double(ncells);
    
    t = 0.;
    tx = 0.;
    ty = 0.;
    for(int i=0;i<ncells;i++){
      int vecino3, vecino4; 
      {
	int fy = i / mx;
	int fx = i % mx;
	int fyp1, fxp1;
	
	fyp1 = ((fy+1) % my) * mx;
	fxp1 = ((fx+1) % mx);
	
	
	fy = fy * mx;
	
	vecino3 = fy   + fxp1;
	vecino4 = fyp1 + fx;
      }
      
      
      tx = tx + 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * (cvx[i] - vmedx)*(cvx[i] - vmedx);
      ty = ty + 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * (cvy[i] - vmedy)*(cvy[i] - vmedy);
      
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
    }
    
    tx = tx/double(ncells);
    ty = ty/double(ncells);
    
    t = (tx + ty)/2.;
    
    sigma_t_fluid = sigma_t_fluid + (t - temperature)*(t - temperature);
    
    t_mean_fluid = t_mean_fluid + t;
    
    txm += tx;
    tym += ty;
    
    cout.precision(15);
    cout << "MASS FLUID " << massTotal << endl;
    cout.precision(6);
    cout << "TEMPERATURE FLUID " << t << " " << t_mean_fluid/double(n) << " " << (t_mean_fluid/double(n) - temperature)/temperature << " " << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    cout << "tx                " << tx << " " << txm/double(n) << endl;
    cout << "ty                " << ty << " " << tym/double(n) << endl;

    if(setparticles==1){
      tx=0;
      ty=0;
      massTotal = mass + volumeParticle * cVolume * densfluid;
      for(int i=0;i<np;i++){
	pxP += mass * vxParticle[i];
	pyP += mass * vyParticle[i];
        tx += massTotal * vxParticle[i] * vxParticle[i];
	ty += massTotal * vyParticle[i] * vyParticle[i];
      }
      tx /= np;
      ty /= np;
      t_mean_particle += (tx+ty)/2.;
      txmP += tx;
      tymP += ty;
      
      /*cout << "TEMPERATURE PARTICLE " << (tx+ty+tz)/np << " " 
	   << t_mean_particle/double(n) << " " 
	   << (t_mean_particle/double(n) - temperature)/temperature << endl;*/
      cout << "TEMPERATURE PARTICLE " << (tx+ty) << " " 
	   << t_mean_particle/double(n) << " " 
	   << (t_mean_particle/double(n) - temperature)/temperature << endl;
      cout << "tx                " << tx << " " << txmP/double(n) << endl;
      cout << "ty                " << ty << " " << tymP/double(n) << endl;
      
    }

    px = pxF + pxP;
    py = pyF + pyP;
    cout.precision(15);
    //cout << "PX " << px << "  " << pxF << "  " << pxP << endl;
    //cout << "PY " << py << "  " << pyF << "  " << pyP << endl << endl;
    cout << "PX " << px << endl;
    cout << "PY " << py << endl << endl;

    cout.precision(6);
  }
  else if(index==2){
    n = step / samplefreq;
    string NombreTime;
    NombreTime =  outputname + ".temperature";
    ofstream fileSave(NombreTime.c_str());
    fileSave.precision(15);
    
    fileSave << "TEMPERATURE FLUID " << t_mean_fluid/double(n) << " " << (t_mean_fluid/double(n) - temperature)/temperature << " " << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    fileSave << "tx                " << txm/double(n) << endl;
    fileSave << "ty                " << tym/double(n) << endl;

    massTotal = 0;
    pxF=0; pyF=0; 
    for(int i=0;i<ncells;i++){
      int vecino3, vecino4; 
      {
	int fy = i / mx;
	int fx = i % mx;
	int fyp1, fxp1;
	
	fyp1 = ((fy+1) % my) * mx;
	fxp1 = ((fx+1) % mx);
	
	
	fy = fy * mx;
	
	vecino3 = fy   + fxp1;
	vecino4 = fyp1 + fx;
      }
      
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
      
      massTotal += cVolume * cDensity[i];
    }
    if(setparticles==1){
      tx=0;
      ty=0;
      massTotal = mass + volumeParticle * cVolume * densfluid;
      for(int i=0;i<np;i++){
        pxP += mass * vxParticle[i];
        pyP += mass * vyParticle[i];
        tx += massTotal * vxParticle[i] * vxParticle[i];
	ty += massTotal * vyParticle[i] * vyParticle[i];
      }
      t_mean_particle += 0.5*(tx+ty)/np;
      fileSave << "TEMPERATURE PARTICLE " << (tx+ty)/np << " " << t_mean_particle/double(n) << " " << (t_mean_particle/double(n) - temperature)/temperature << endl;
      fileSave << "tx                " << tx << " " << txmP/double(n) << endl;
      fileSave << "ty                " << ty << " " << tymP/double(n) << endl;
    }

    px = pxF + pxP;
    py = pyF + pyP;
    fileSave.precision(15);
    fileSave << "PX " << px << endl;
    fileSave << "PY " << py << endl;
    fileSave.precision(6);

    fileSave << "MASS FLUID " << massTotal << endl;
    
    fileSave.close();      
    
  }
  
  
  
  
  
  return 1;
  
}  
