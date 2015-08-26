// Filename: temperatureGhost.cpp
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



bool temperatureGhost(int index){

  int n;
  double pxF, pyF, pzF;
  double massTotal; 
  
  
  double t, vmedx, vmedy, vmedz;
  double tx, ty, tz;
  
  if(index==0){
    return 1;
  }
  else if(index==1){
    n = step / samplefreq;
    
    vmedx = 0.;
    vmedy = 0.;
    vmedz = 0.;
    
    massTotal = 0;
    pxF=0; pyF=0; pzF=0;
    
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
      vmedx = vmedx + cvx[i]; 
      vmedy = vmedy + cvy[i];
      vmedz = vmedz + cvz[i];
      massTotal += cVolume * cDensity[i];
    }
    
    vmedx = vmedx/double(ncells);
    vmedy = vmedy/double(ncells);
    vmedz = vmedz/double(ncells);
    
    t = 0.;
    tx = 0.;
    ty = 0.;
    tz = 0.;
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
      int vecino3, vecino4, vecino5; 
      vecino3 = i+1;
      vecino4 = i+mxt;
      vecino5 = i+mxt*myt;
      
      tx = tx + 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * (cvx[i] - vmedx)*(cvx[i] - vmedx);
      ty = ty + 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * (cvy[i] - vmedy)*(cvy[i] - vmedy);
      tz = tz + 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * (cvz[i] - vmedz)*(cvz[i] - vmedz);
      
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
      pzF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * cvz[i];
    }
    
    tx = tx/double(ncells);
    ty = ty/double(ncells);
    tz = tz/double(ncells);
    
    t = (tx + ty + tz)/3.;
    
    sigma_t_fluid = sigma_t_fluid + (t - temperature)*(t - temperature);
    
    t_mean_fluid = t_mean_fluid + t;
    
    txm += tx;
    tym += ty;
    tzm += tz;
    cout.precision(15);    
    cout << "MASS FLUID " << massTotal << endl;
    cout.precision(6);
    cout << "TEMPERATURE FLUID " << t << " " << t_mean_fluid/double(n) << " " << (t_mean_fluid/double(n) - temperature)/temperature << " " << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    cout << "tx                " << tx << " " << txm/double(n) << endl;
    cout << "ty                " << ty << " " << tym/double(n) << endl;
    cout << "tz                " << tz << " " << tzm/double(n) <<  endl;
    
    
    cout << "PX " << pxF << endl;
    cout << "PY " << pyF << endl;
    cout << "PZ " << pzF << endl << endl;;
  }
  else if(index==2){
    massTotal = 0;
    pxF=0; pyF=0; pzF=0;    
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
      int vecino3, vecino4, vecino5; 
      vecino3 = i+1;
      vecino4 = i+mxt;
      vecino5 = i+mxt*myt;

      massTotal += cVolume * cDensity[i];
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
      pzF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * cvz[i];
    }


    n = step / samplefreq;
    string NombreTime;
    NombreTime =  outputname + ".temperature";
    ofstream fileSave(NombreTime.c_str());
    fileSave.precision(15);
    fileSave << "MASS FLUID " << massTotal << endl;
    fileSave << "TEMPERATURE FLUID " << t_mean_fluid/double(n) << " " << (t_mean_fluid/double(n) - temperature)/temperature << " " << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    fileSave << "tx                " << txm/double(n) << endl;
    fileSave << "ty                " << tym/double(n) << endl;
    fileSave << "tz                " << tzm/double(n) <<  endl;
    
    
    fileSave << "PX " << pxF << endl;
    fileSave << "PY " << pyF << endl;
    fileSave << "PZ " << pzF << endl << endl;;
    fileSave << "totalMass " << massTotal << endl << endl;
    
    fileSave.close();      
    
  }
  
  

  
  
  return 1;
  
}  
