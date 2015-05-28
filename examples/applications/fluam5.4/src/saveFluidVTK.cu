// Filename: saveFluidVTK.cu
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


#include <sstream>
#include "header.h"
#include "headerOtherFluidVariables.h"
#include "fluid.h"
#include "cells.h"
#include "particles.h"
#include "visit_writer.h"
#include "visit_writer.c"
#include "hydroAnalysis.h"


bool saveFluidVTK(int option){
  
  string savefile;
  ofstream file;
  
  if(option==0){
    //Save snapshot by index
    int index = step/savefreq ;
    stringstream s;
    s << index;
    string st = s.str();
    savefile = outputname + "." + st + ".fluid.vtk";


  }
  else if(option==1){
    //Save final snapshot without index  
    //And free memory at the end
    savefile = outputname + ".fluid.vtk";
  }
  else{
    //ERROR
    return 0;
  }
  


  //Save in VTK format
  //See manual "Getting Data into Visit"
  file.open(savefile.c_str());  
  
  //Vector for the velocity
  double *vectorVelocities; 
  //vectorVelocities = new double [3*mx*my*mz]; //Use this for 3D
  vectorVelocities = new double [3*mx*mz]; //Use this for projection in 2D

  int nvars;
  int *vardims;
  int *centering;
  char **varnames;
  double **vars;

  char nameVelocity[] = {"velocity"};
  char nameDensity[] = {"density"};
  char nameConcentration[] = {"concentration"};

  //Determine number of variables
  if(setContinuousGradient ||
     setGiantFluctuations  ||
     setBinaryMixtureWall  ||
     setBinaryMixture){
    nvars = 3; //velocity, density and concentration
    vardims = new int [3];
    vardims[0] = 3;
    vardims[1] = 3;
    vardims[2] = 3;
    centering = new int [3];
    centering[0] = 0;
    centering[1] = 0;
    centering[2] = 0;
    varnames = new char* [3];
    varnames[0] = &nameVelocity[0];
    varnames[1] = &nameDensity[0];
    varnames[2] = &nameConcentration[0];
    vars = new double* [3];
    vars[0] = vectorVelocities;
    vars[1] = cDensity;
    vars[2] = c;
  }
  else if(thermostat ||
	  setGhost){
    nvars = 2; //velocity and density
    vardims = new int[2];
    vardims[0] = 3;
    vardims[1] = 1;
    centering = new int [2];
    centering[0] = 0;
    centering[1] = 0;
    varnames = new char* [2];
    varnames[0] = &nameVelocity[0];
    varnames[1] = &nameDensity[0];
    vars = new double* [2];
    vars[0] = vectorVelocities;
    vars[1] = cDensity;
  }
  else if(incompressibleBinaryMixture ||
	  incompressibleBinaryMixtureMidPoint){
    nvars = 2; //velocity and concentration
    vardims = new int[2];
    vardims[0] = 3;
    vardims[1] = 1;
    centering = new int [2];
    centering[0] = 0;
    centering[1] = 0;
    varnames = new char* [2];
    varnames[0] = &nameVelocity[0];
    varnames[1] = &nameConcentration[0];
    vars = new double* [2];
    vars[0] = vectorVelocities;
    vars[1] = c;
  }
  else{
    nvars = 1; //velocity
    vardims = new int[1];
    vardims[0] = 3;
    centering = new int [1];
    centering[0] = 0;
    varnames = new char* [1];
    varnames[0] = &nameVelocity[0];
    vars = new double* [1];
    vars[0] = vectorVelocities;
  }
     
  //int dims[] = {mx+1, my+1, mz+1};
  int dims[] = {mx+1, mz+1, 1}; //Use this for projection in 2D


  //For vtk the velovity is in the
  //center of each cell, we need to
  //interpolate
  if(setparticles==0){

    for(int i=0;i<ncells;i++){
      int fz = i/(mx*my);
      int fy = (i % (mx*my))/mx;
      int fx = i % mx;
      
      int fzm1 = ((fz-1+mz) % mz) ;
      int fym1 = ((fy-1+my) % my) ;
      int fxm1 = ((fx-1+mx) % mx);
      
      
      fz = fz ;
      fy = fy ;
      
      int vecino2 = fz*mx*my   + fy*mx   + fxm1;
      int vecino1 = fz*mx*my   + fym1*mx + fx;
      int vecino0 = fzm1*mx*my + fy*mx   + fx;    
      
      
      
      vectorVelocities[3*i]   = 0.5*(cvx[vecino2]+cvx[i]) ;
      vectorVelocities[1+3*i] = 0.5*(cvy[vecino1]+cvy[i]) ;
      vectorVelocities[2+3*i] = 0.5*(cvz[vecino0]+cvz[i]) ;
    }
  }
  else{
    //Center the system on the particle 0
    double r;
    r = rxParticle[0];
    r = r - (int(r/lx + 0.5*((r>0)-(r<0)))) * lx;
    int kxP = int(r/lx*mx + 0.5*mx) % mx;
    r = ryParticle[0];
    r = r - (int(r/ly + 0.5*((r>0)-(r<0)))) * ly;
    int kyP = int(r/ly*my + 0.5*my) % my;
    r = rzParticle[0];
    r = r - (int(r/lz + 0.5*((r>0)-(r<0)))) * lz;
    int kzP = int(r/lz*mz + 0.5*mz) % mz;
    

    for(int i=0;i<ncells;i++){
      int fz = i/(mx*my);
      int fy = (i % (mx*my))/mx;
      int fx = i % mx;
      

    
      int fzm1 = ((fz-1+mz) % mz) ;
      int fym1 = ((fy-1+my) % my) ;
      int fxm1 = ((fx-1+mx) % mx);
            
      
      int vecino2 = fz*mx*my      + fy*mx      + fxm1;
      int vecino1 = fz*mx*my      + fym1*mx    + fx;
      int vecino0 = fzm1*mx*my    + fy*mx      + fx;    
      

      int kxNew = (fx - kxP + mx/2 + mx) % mx;
      int kyNew = (fy - kyP + my/2 + my) % my;
      int kzNew = (fz - kzP + mz/2 + mz) % mz;

      if(fy!=kyP) continue; //Use this for projection in 2D
      int j = kxNew + kzNew*mx; //Use this for projection in 2D
      //int j = kxNew + kyNew*mx + kzNew*mx*my;
      
      //vectorVelocities[3*j]   = 0.5*(cvx[vecino2]+cvx[i]) - vxParticleI[0];
      //vectorVelocities[1+3*j] = 0.5*(cvy[vecino1]+cvy[i]) - vyParticleI[0];
      //vectorVelocities[2+3*j] = 0.5*(cvz[vecino0]+cvz[i]) - vzParticleI[0];
      vectorVelocities[3*j]   = 0.5*(cvx[vecino2]+cvx[i]) - vxParticleI[0];//Use this for projection in 2D
      vectorVelocities[1+3*j] = 0.5*(cvz[vecino0]+cvz[i]) - vzParticleI[0];//Use this for projection in 2D
      vectorVelocities[2+3*j] = 0.5*(cvy[vecino1]+cvy[i]) - vyParticleI[0];//Use this for projection in 2D
    }
  }


  /*Use visit_writer to write a regular mesh with data. */
  write_regular_mesh(savefile.c_str(), //Output file
		     0, //0=ASCII,  1=Binary
		     dims, // {mx, my, mz}
		     nvars, // number of variables
		     vardims, // Size of each variable, 1=scalar, velocity=3*scalars
		     centering, // 
		     varnames, //
		     vars //
		     );

  //Free memory
  delete[] vectorVelocities;  
  delete[] vardims;
  delete[] centering;
  delete[] varnames;
  delete[] vars;

  
  file.close();
    






  return 1;

}
