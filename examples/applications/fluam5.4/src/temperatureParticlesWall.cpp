// Filename: temperatureParticlesWall.cpp
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
#include "headerOtherFluidVariables.h"
#include "fluid.h"
#include "temperature.h"
#include "cells.h"
#include "particles.h"


bool temperatureParticlesWall(int index){

  int n;
  double px, py, pz;
  double pxF, pyF, pzF;
  double pxP, pyP, pzP;
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
    pxP=0; pyP=0; pzP=0;
    
    for(int j=0;j<ncells;j++){
      int i, fx, fy, fz;
      fx = j % mx;
      fy = (j % (mx*my)) / mx;
      fz = j / (mx*my);
      //go to index tanking into account ghost cells
      fy++;
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
      fy++;
      
      i = fx + fy*mxt + fz*mxt*myt;

      int vecino3, vecino4, vecino5; 
      {
	int mxmy = mx * myt;
	//int fz = i/mxmy;
	//int fy = (i % mxmy)/mx;
	//int fx = i % mx;
	int fzp1, fyp1, fxp1;
	
	fzp1 = ((fz+1) % mz) * mxmy;
	fyp1 = ((fy+1) % myt) * mx;
	fxp1 = ((fx+1) % mx);
	
	
	fz = fz * mxmy;
	fy = fy * mx;
	
	vecino3 = fz   + fy   + fxp1;
	vecino4 = fz   + fyp1 + fx;
	vecino5 = fzp1 + fy   + fx;    

      }


      
      tx = tx + 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * (cvx[i] - vmedx)*(cvx[i] - vmedx);
      ty = ty + 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * (cvy[i] - vmedy)*(cvy[i] - vmedy);
      tz = tz + 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * (cvz[i] - vmedz)*(cvz[i] - vmedz);
      
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
      pzF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * cvz[i];
    }
    
    tx = tx/double(ncells);
    ty = ty/double(ncells-mx*mz);
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
    cout << "TEMPERATURE FLUID " << t << " " << t_mean_fluid/double(n) << " " 
	 << (t_mean_fluid/double(n) - temperature)/temperature << " " 
	 << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    cout << "tx                " << tx << " " << txm/double(n) << endl;
    cout << "ty                " << ty << " " << tym/double(n) << endl;
    cout << "tz                " << tz << " " << tzm/double(n) <<  endl;
    
    if(setparticles==1){
      tx=0;
      ty=0;
      tz=0;
      massTotal = mass + volumeParticle * cVolume * densfluid;
      for(int i=0;i<np;i++){
	pxP += mass * vxParticle[i];
	pyP += mass * vyParticle[i];
        pzP += mass * vzParticle[i];
        tx += massTotal * vxParticle[i] * vxParticle[i]/3.;
	ty += massTotal * vyParticle[i] * vyParticle[i]/3.;
        tz += massTotal * vzParticle[i] * vzParticle[i]/3.;
      }
      t_mean_particle += (tx+ty+tz)/np;
      cout << "TEMPERATURE PARTICLE " << (tx+ty+tz)/np << " " 
	   << t_mean_particle/double(n) << " " 
	   << (t_mean_particle/double(n) - temperature)/temperature << endl;
    }

    px = pxF + pxP;
    py = pyF + pyP;
    pz = pzF + pzP;
    cout.precision(15);
    cout << "PX " << px << endl;
    cout << "PY " << py << endl;
    cout << "PZ " << pz << endl << endl;;
    cout.precision(6);

  }
  else if(index==2){
    massTotal = 0;
    pxF=0; pyF=0; pzF=0;    
    pxP=0; pyP=0; pzP=0;
    for(int j=0;j<ncells;j++){
      int i, fx, fy, fz;
      fx = j % mx;
      fy = (j % (mx*my)) / mx;
      fz = j / (mx*my);
      //go to index tanking into account ghost cells
      fy++;
      i = fx + fy*mxt + fz*mxt*myt;
      

      int vecino3, vecino4, vecino5; 
      {
	int mxmy = mx * myt;
	int fz = i/mxmy;
	int fy = (i % mxmy)/mx;
	int fx = i % mx;
	int fzp1, fyp1, fxp1;
	
	fzp1 = ((fz+1) % mz) * mxmy;
	fyp1 = ((fy+1) % myt) * mx;
	fxp1 = ((fx+1) % mx);
	
	
	fz = fz * mxmy;
	fy = fy * mx;
	
	vecino3 = fz   + fy   + fxp1;
	vecino4 = fz   + fyp1 + fx;
	vecino5 = fzp1 + fy   + fx;    
      }



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
    fileSave << "TEMPERATURE FLUID " << t_mean_fluid/double(n) 
	     << " " << (t_mean_fluid/double(n) - temperature)/temperature 
	     << " " << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    fileSave << "tx                " << txm/double(n) << endl;
    fileSave << "ty                " << tym/double(n) << endl;
    fileSave << "tz                " << tzm/double(n) <<  endl;
    
    
    if(setparticles==1){
      tx=0;
      ty=0;
      tz=0;
      massTotal = mass + volumeParticle * cVolume * densfluid;
      for(int i=0;i<np;i++){
        pxP += mass * vxParticle[i];
        pyP += mass * vyParticle[i];
        pzP += mass * vzParticle[i];
        tx += massTotal * vxParticle[i] * vxParticle[i]/3.;
	ty += massTotal * vyParticle[i] * vyParticle[i]/3.;
        tz += massTotal * vzParticle[i] * vzParticle[i]/3.;
      }
      t_mean_particle += (tx+ty+tz)/np;
      fileSave << "TEMPERATURE PARTICLE " << (tx+ty+tz)/np << " " 
	       << t_mean_particle/double(n) << " " 
	       << (t_mean_particle/double(n) - temperature)/temperature << endl;
    }

    px = pxF + pxP;
    py = pyF + pyP;
    pz = pzF + pzP;
    fileSave.precision(15);
    fileSave << "PX " << px  << endl;
    fileSave << "PY " << py  << endl;
    fileSave << "PZ " << pz  << endl << endl;;
    fileSave << "totalMass " << massTotal << endl << endl;

    fileSave.precision(6);    
    fileSave.close();      
    
  }
  
  

  
  
  return 1;
  
}  


























bool temperatureParticlesWall2(int index){

  int n;
  double px, py, pz;
  double pxF, pyF, pzF;
  double pxP, pyP, pzP;
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
    px=0; py=0; pz=0;
    pxF=0; pyF=0; pzF=0;
    pxP=0; pyP=0; pzP=0;
    
    
    for(int i=0;i<ncellst;i++){
      vmedx = vmedx + cvx[i]; 
      vmedy = vmedy + cvy[i];
      vmedz = vmedz + cvz[i];
      massTotal += cVolume * cDensity[i];
    }
    
    vmedx = vmedx/double(ncellst);
    vmedy = vmedy/double(ncellst);
    vmedz = vmedz/double(ncellst);
    
    t = 0.;
    tx = 0.;
    ty = 0.;
    tz = 0.;
    for(int i=0;i<ncellst;i++){
      int vecino3, vecino4, vecino5; 
      {
	int mxmy = mx * myt;
	int fz = i/mxmy;
	int fy = (i % mxmy)/mx;
	int fx = i % mx;
	int fzp1, fyp1, fxp1;
	
	fzp1 = ((fz+1) % mz) * mxmy;
	fyp1 = ((fy+1) % myt) * mx;
	fxp1 = ((fx+1) % mx);
	
	
	fz = fz * mxmy;
	fy = fy * mx;
	
	vecino3 = fz   + fy   + fxp1;
	vecino4 = fz   + fyp1 + fx;
	vecino5 = fzp1 + fy   + fx;    
      }
      
      
      tx = tx + 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * (cvx[i] - vmedx)*(cvx[i] - vmedx);
      ty = ty + 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * (cvy[i] - vmedy)*(cvy[i] - vmedy);
      tz = tz + 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * (cvz[i] - vmedz)*(cvz[i] - vmedz);
      
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
      pzF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * cvz[i];
    }
    
    tx = tx/double(ncellst);
    ty = ty/double(ncellst);
    tz = tz/double(ncellst);
    
    t = (tx + ty + tz)/3.;
    
    sigma_t_fluid = sigma_t_fluid + (t - temperature)*(t - temperature);
    
    t_mean_fluid = t_mean_fluid + t;
    
    txm += tx;
    tym += ty;
    tzm += tz;
    
    cout << "MASS FLUID " << massTotal << endl;
    cout << "TEMPERATURE FLUID " << t << " " << t_mean_fluid/double(n) << " " << (t_mean_fluid/double(n) - temperature)/temperature << " " << sqrt(sigma_t_fluid/double(n))/temperature << "  " << n << endl;
    
    cout << "tx                " << tx << " " << txm/double(n) << endl;
    cout << "ty                " << ty << " " << tym/double(n) << endl;
    cout << "tz                " << tz << " " << tzm/double(n) <<  endl;

    if(setparticles==1){
      tx=0;
      ty=0;
      tz=0;
      massTotal = mass + volumeParticle * cVolume * densfluid;
      for(int i=0;i<np;i++){
	pxP += mass * vxParticle[i];
	pyP += mass * vyParticle[i];
        pzP += mass * vzParticle[i];
        tx += massTotal * vxParticle[i] * vxParticle[i]/3.;
	ty += massTotal * vyParticle[i] * vyParticle[i]/3.;
        tz += massTotal * vzParticle[i] * vzParticle[i]/3.;
      }
      t_mean_particle += (tx+ty+tz)/np;
      cout << "TEMPERATURE PARTICLE " << (tx+ty+tz)/np << " " 
	   << t_mean_particle/double(n) << " " 
	   << (t_mean_particle/double(n) - temperature)/temperature << endl;
    }

    px = pxF + pxP;
    py = pyF + pyP;
    pz = pzF + pzP;
    cout.precision(15);
    cout << "PXF " << pxF << "    " << " PXP " << pxP << endl;
    cout << "PYF " << pyF << "    " << " PYP " << pyP << endl;
    cout << "PZF " << pzF << "    " << " PZP " << pzP << endl << endl;;
    cout << "PX " << px << endl;
    cout << "PY " << py << endl;
    cout << "PZ " << pz << endl << endl;;
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
    fileSave << "tz                " << tzm/double(n) <<  endl;

    massTotal = 0;
    pxF=0; pyF=0; pzF=0;
    for(int i=0;i<ncellst;i++){
      int vecino3, vecino4, vecino5; 
      {
	int mxmy = mx * myt;
	int fz = i/mxmy;
	int fy = (i % mxmy)/mx;
	int fx = i % mx;
	int fzp1, fyp1, fxp1;
	
	fzp1 = ((fz+1) % mz) * mxmy;
	fyp1 = ((fy+1) % myt) * mx;
	fxp1 = ((fx+1) % mx);
	
	
	fz = fz * mxmy;
	fy = fy * mx;
	
	vecino3 = fz   + fy   + fxp1;
	vecino4 = fz   + fyp1 + fx;
	vecino5 = fzp1 + fy   + fx;    
      }
      
      pxF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino3]) * cvx[i];
      pyF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino4]) * cvy[i];
      pzF += 0.5 * cVolume * (cDensity[i] + cDensity[vecino5]) * cvz[i];
      
      massTotal += cVolume * cDensity[i];
    }
    if(setparticles==1){
      tx=0;
      ty=0;
      tz=0;
      massTotal = mass + volumeParticle * cVolume * densfluid;
      for(int i=0;i<np;i++){
        pxP += mass * vxParticle[i];
        pyP += mass * vyParticle[i];
        pzP += mass * vzParticle[i];
        tx += massTotal * vxParticle[i] * vxParticle[i]/3.;
	ty += massTotal * vyParticle[i] * vyParticle[i]/3.;
        tz += massTotal * vzParticle[i] * vzParticle[i]/3.;
      }
      t_mean_particle += (tx+ty+tz)/np;
      fileSave << "TEMPERATURE PARTICLE " << (tx+ty+tz)/np << " " << t_mean_particle/double(n) << " " << (t_mean_particle/double(n) - temperature)/temperature << endl;
    }

    px = pxF + pxP;
    py = pyF + pyP;
    pz = pzF + pzP;
    fileSave.precision(15);
    fileSave << "PX " << px << endl;
    fileSave << "PY " << py << endl;
    fileSave << "PZ " << pz << endl << endl;;
    fileSave.precision(6);

    fileSave << "MASS FLUID " << massTotal << endl;
    //fileSave << "totalMass " << massTotal << endl << endl;
    
    fileSave.close();      
    
  }
  
  
  
  
  
  return 1;
  
}  
