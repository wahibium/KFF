// Filename: initializeBondedForces.cu
//
// Copyright (c) 2010-2012, Florencio Balboa Usabiaga
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



//NEW bonded forces


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



bool initializeBondedForces(){
  
  int index1, index2, indexOld;
  int nParticleParticleMemory=0;
  int nParticleFixedPointMemory=0;
  int trashInt;
  double trashDouble;
  indexOld=-1;


  //OPEN FILE
  ifstream file(bondedForcesFile.c_str());

  //Number of particle-particle bonds
  //IMPORTANT, each bonds should be count twice
  file >> nbondsParticleParticle;
  
  //Allocate memory
  bondsParticleParticle = new int [np];
  bondsParticleParticleOffset = new int [np];

  //Initially any particle has bonds
  for(int i=0;i<np;i++)
    bondsParticleParticle[i] = 0;

  //Information bonds particle-particle
  for(int i=0;i<nbondsParticleParticle;i++){  
    file >> index1 >> index2 >> trashDouble >> trashDouble;
    //cout << "AAA " << index1 << "   " << index2 << "   " << trashDouble << endl << endl << endl;
    if(index1<indexOld){
      cout << "ERROR, bad sorting in  bonded Forces Particle-Particle" << endl;
      return 0;
    }
    else if(index1!=indexOld){
      bondsParticleParticleOffset[index1]=nParticleParticleMemory;
      //nParticleParticleMemory++;//size for the array bondsIndexParticleParticle
    }
    nParticleParticleMemory++;
    bondsParticleParticle[index1]++;
    indexOld=index1;
  }






  


  indexOld=-1;

  //Number of particle-fixedPoints bonds
  //IMPORTANT, each bonds should be count once
  file >> nbondsParticleFixedPoint;

  //cout << "BBB " << nbondsParticleFixedPoint << endl << endl << endl;

  //Allocate memory
  bondsParticleFixedPoint = new int [np];
  bondsParticleFixedPointOffset = new int [np];

  //Initially any particle has bonds
  for(int i=0;i<np;i++)
    bondsParticleFixedPoint[i] = 0;

  //Information bonds particle-fixedPoint
  for(int i=0;i<nbondsParticleFixedPoint;i++){  
    file >> index1 >> trashDouble >> trashDouble >>
      trashDouble >> trashDouble >> trashDouble;
    //cout << "CCC " << index1 << endl << endl << endl;
    if(index1<indexOld){
      cout << "ERROR, bad sorting in  bonded Forces Particle-fixedPoint" << endl;
      return 0;
    }
    else if(index1!=indexOld){
      bondsParticleFixedPointOffset[index1]=nParticleFixedPointMemory;
      //nParticleFixedPointMemory++;
    }
    nParticleFixedPointMemory++;
    bondsParticleFixedPoint[index1]++;
    index1=indexOld;
  }

  //CLOSE FILE
  file.close();


  //Allocate memory
  bondsIndexParticleParticle = new int [nbondsParticleParticle];
  kSpringParticleParticle = new double [nbondsParticleParticle];
  r0ParticleParticle = new double [nbondsParticleParticle];


  //bondsIndexParticleFixedPoint = new int [nbondsParticleFixedPoint];  
  kSpringParticleFixedPoint = new double [nbondsParticleFixedPoint];
  r0ParticleFixedPoint = new double [nbondsParticleFixedPoint];
  rxFixedPoint = new double [nbondsParticleFixedPoint];
  ryFixedPoint = new double [nbondsParticleFixedPoint];
  rzFixedPoint = new double [nbondsParticleFixedPoint];




  //OPEN THE FILE AGAIN
  file.open(bondedForcesFile.c_str());



  //Number of particle-particle bonds
  //IMPORTANT, each bonds should be count twice
  file >> nbondsParticleParticle;

  //cout << "DDD " << nbondsParticleParticle << endl << endl << endl;

  //Information bonds particle-particle
  int n=0;
  indexOld=-1;
  double a, b;
  for(int i=0;i<nbondsParticleParticle;i++){  
    //file >> index1 >> index2 >> a >> b ;
    /*bondsIndexParticleParticle[bondsParticleParticleOffset[index1]+n] = index2;
    kSpringParticleParticle[   bondsParticleParticleOffset[index1]+n] = a;
    r0ParticleParticle[        bondsParticleParticleOffset[index1]+n] = b;*/

    //cout << "EEE " << index1 << "   " << index2 << "   " << a << "   " << b << endl;
    
    file >> index1;
    if(index1==indexOld){
      n++;
    }
    else{
      n=0;
    }
    file >> bondsIndexParticleParticle[bondsParticleParticleOffset[index1]+n]
	 >> kSpringParticleParticle[             bondsParticleParticleOffset[index1]+n]
	 >> r0ParticleParticle[                  bondsParticleParticleOffset[index1]+n];
 
    /*cout << "FFF " << index1 << "   " 
	 << bondsIndexParticleParticle[bondsParticleParticleOffset[index1]+n] << "   "
	 << kSpringParticleParticle[             bondsParticleParticleOffset[index1]+n] << "   "
	 << r0ParticleParticle[                  bondsParticleParticleOffset[index1]+n] << "   "
	 << endl << endl;*/
    indexOld=index1;
  }



  //Number of particle-fixedPoints bonds
  //IMPORTANT, each bonds should be count once
  file >> nbondsParticleFixedPoint;

  //Information bonds particle-fixedPoint
  n=0;
  indexOld=-1;
  for(int i=0;i<nbondsParticleFixedPoint;i++){  
    file >> index1;
    if(index1==indexOld){
      n++;
    }
    else{
      n=0;
    }
    file >> kSpringParticleFixedPoint[             bondsParticleFixedPointOffset[index1]+n]
	 >> r0ParticleFixedPoint[                  bondsParticleFixedPointOffset[index1]+n]
	 >> rxFixedPoint[                          bondsParticleFixedPointOffset[index1]+n]
	 >> ryFixedPoint[                          bondsParticleFixedPointOffset[index1]+n]
	 >> rzFixedPoint[                          bondsParticleFixedPointOffset[index1]+n];

    indexOld=index1;
  }




  //CLOSE FILE
  file.close();

  cout << "nParticleParticeMemory    " << nParticleParticleMemory << endl;
  /*for(int i=0;i<np;i++){
    cout << "Particle     " << i << endl;
    cout << "number bonds " << bondsParticleParticle[i] << endl;
    cout << "offset       " << bondsParticleParticleOffset[i] << endl;
    for(int j=0;j<bondsParticleParticle[i];j++){
      cout << "link " << j << " index "<<bondsIndexParticleParticle[bondsParticleParticleOffset[i] + j]<<endl;
      cout << "link " << j << " k     "<<kSpringParticleParticle[bondsParticleParticleOffset[i] + j] << endl;
      cout << "link " << j << " r0    "<< r0ParticleParticle[bondsParticleParticleOffset[i] + j] << endl; 
    }
    cout << endl << endl;
    }*/




  cout << "INITALIZE BONDED FORCES :       DONE " << endl;

  return 1;
}















bool freeBondedForces(){

  delete[] bondsParticleParticle;
  delete[] bondsParticleFixedPoint;

  delete[] bondsParticleParticleOffset;
  delete[] bondsParticleFixedPointOffset;


  delete[] bondsIndexParticleParticle;
  delete[] kSpringParticleParticle;
  delete[] r0ParticleParticle;
  

  //delete[] bondsIndexParticleFixedPoint;
  delete[] kSpringParticleFixedPoint;
  delete[] r0ParticleFixedPoint;
  delete[] rxFixedPoint;
  delete[] ryFixedPoint;
  delete[] rzFixedPoint;
  
}
