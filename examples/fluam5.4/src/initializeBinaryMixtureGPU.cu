// Filename: initializeBinaryMixtureGPU.cu
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


bool initializeBinaryMixtureGPU(){

  cutilSafeCall(cudaMalloc((void**)&cGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&cPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&dcGPU,ncells*sizeof(double)));

  cutilSafeCall(cudaMemcpy(cGPU,c,ncells*sizeof(double),cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMemcpyToSymbol(diffusionGPU,&diffusion,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(massSpecies0GPU,&massSpecies0,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(massSpecies1GPU,&massSpecies1,sizeof(double)));



  cout << "INITIALIZE BINARY MIXTURE GPU : DONE" << endl;
  return 1;
}
