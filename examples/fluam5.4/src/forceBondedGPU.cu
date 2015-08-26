// Filename: forceBondedGPU.cu
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


__device__ void forceBondedParticleParticleGPU(const int i,
					       double& fx, //Pass by reference
					       double& fy,
					       double& fz,
					       const double rx,
					       const double ry,
					       const double rz,
					       const bondedForcesVariables* bFV){	

  double x, y, z;
  double r, r0;
  double kSpring;
  int index;


  //Particle-Particle Force
  int nBonds = bFV->bondsParticleParticleGPU[i];
  int offset = bFV->bondsParticleParticleOffsetGPU[i];
  
  
  for(int j=0;j<nBonds;j++){

    index = bFV->bondsIndexParticleParticleGPU[offset + j];

    //if(i==0) index=1;
    //if(i==1) index=0;


    //Particle bonded coordinates
    x = fetch_double(texrxboundaryGPU,nboundaryGPU+index);
    y = fetch_double(texryboundaryGPU,nboundaryGPU+index);
    z = fetch_double(texrzboundaryGPU,nboundaryGPU+index);
    
    //Equilibrium distance 
    r0 = bFV->r0ParticleParticleGPU[offset+j];
    
    //Spring constant
    kSpring = bFV->kSpringParticleParticleGPU[offset+j];
    
    if(r0==0){
      fx += -kSpring * (rx - x);
      fy += -kSpring * (ry - y);
      fz += -kSpring * (rz - z);
    }  
    else{     //If r0!=0 calculate particle particle distance
      r = sqrt( (x-rx)*(x-rx) + (y-ry)*(y-ry) + (z-rz)*(z-rz) );
      if(r>0){//If r=0 -> f=0
	fx += -kSpring * (1 - r0/r) * (rx - x);
	fy += -kSpring * (1 - r0/r) * (ry - y);
	fz += -kSpring * (1 - r0/r) * (rz - z);
      }
    }
  }











  //Particle-FixedPoint Force
  nBonds = bFV->bondsParticleFixedPointGPU[i];
  offset = bFV->bondsParticleFixedPointOffsetGPU[i];
  
  
  for(int j=0;j<nBonds;j++){
    

    //Fixed point coordinates
    x = bFV->rxFixedPointGPU[offset+j];
    y = bFV->ryFixedPointGPU[offset+j];
    z = bFV->rzFixedPointGPU[offset+j];
    
    //Equilibrium distance 
    r0 = bFV->r0ParticleFixedPointGPU[offset+j];

    //Spring constant
    kSpring = bFV->kSpringParticleFixedPointGPU[offset+j];
    
    if(r0==0){
      fx += -kSpring * (rx - x);
      fy += -kSpring * (ry - y);
      fz += -kSpring * (rz - z);
    }  
    else{     //If r0!=0 calculate particle particle distance
      r = sqrt( (x-rx)*(x-rx) + (y-ry)*(y-ry) + (z-rz)*(z-rz) );
      if(r>0){//If r=0 -> f=0
	fx += -kSpring * (1 - r0/r) * (rx - x);
	fy += -kSpring * (1 - r0/r) * (ry - y);
	fz += -kSpring * (1 - r0/r) * (rz - z);
      }
    }
  }

    




  return ;
}
