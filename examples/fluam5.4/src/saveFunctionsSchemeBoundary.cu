// Filename: saveFunctionsSchemeBoundary.cu
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


bool saveFunctionsSchemeBoundary(int index, long long step){

  //Initialize save functions
  //cout << "INDEX " << index << endl;
  if(index==0){
    if(!saveSeed()) return 0;
    if(!temperatureBoundary(index)) return 0;
    if(setparticles)
      if(!saveParticles(index,step)) return 0;
    //if(!hydroAnalysis(0)) return 0;
    //if(!covarianceKernel(index)) return 0;
    //if(!saveCellsAlongZ(index)) return 0;
    if(setCheckVelocity==1) checkVelocity(0,0,fileCheckVelocity);
    if(freeEnergyCompressibleParticles || momentumCoupling)
      if(!saveParticlesDensity(0,step)) return 0;
    if(!saveTime(index)) return 0;
  }
  //Use save functions
  else if(index==1){
    if(!temperatureBoundary(index)) return 0;
    if(setparticles)
      if(!saveParticles(index,step)) return 0;
    //if(!hydroAnalysis(1)) return 0;
    //if(!covarianceKernel(index)) return 0;
    //if(!saveCellsAlongZ(index)) return 0;
    if(freeEnergyCompressibleParticles || momentumCoupling)
      if(!saveParticlesDensity(1,step)) return 0;
    if(setCheckVelocity==1) checkVelocity(1,step,"0");
  }
  //Close save functions
  else if(index==2){
    if(!saveTime(index)) return 0;
    if(!temperatureBoundary(index)) return 0;
    if(setparticles)
      if(!saveParticles(index,step)) return 0;
    //if(!hydroAnalysis(2)) return 0;
    //if(!covarianceKernel(index)) return 0;
    //if(!saveCellsAlongZ(index)) return 0;
    if(freeEnergyCompressibleParticles || momentumCoupling)
      if(!saveParticlesDensity(2,step)) return 0;
    if(setCheckVelocity==1) checkVelocity(2,0,fileCheckVelocity);
    if(!saveFluidFinalConfiguration()) return 0;
  }
  else{
    cout << "SAVE FUNCTIONS ERROR, INDEX !=0,1,2 " << endl;
    return 0;
  }
  


  return 1;
}
