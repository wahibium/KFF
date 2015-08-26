// Filename: initializeFluidBinaryMixtureGPU.cu
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


bool initializeFluidBinaryMixtureGPU(){
  
  cudaMemcpy(densityGPU,cDensity,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vxGPU,cvx,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vyGPU,cvy,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vzGPU,cvz,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(cGPU,c,ncellst*sizeof(double),cudaMemcpyHostToDevice);


  cudaMemcpy(rxcellGPU,crx,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(rycellGPU,cry,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(rzcellGPU,crz,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  

  
  
  cout << "INITIALIZE FLUID GPU :          DONE" << endl;

  return 1;
}
