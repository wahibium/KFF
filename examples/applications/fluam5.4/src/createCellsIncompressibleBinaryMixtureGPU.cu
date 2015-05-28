// Filename: createCellsIncompressibleBinaryMixtureGPU.cu
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


#define GPUVARIABLES 1


bool createCellsIncompressibleBinaryMixtureGPU(){
  //Number of cells to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(myGPU,&my,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int)));

  cutilSafeCall(cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int)));

  //Simulation box size to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double)));

  //Time step to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double)));

  //Volume cell to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double)));

  //Viscosity and temperature to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool)));

  //Mass diffusion coefficient to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(diffusionGPU,&diffusion,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(massSpecies0GPU,&massSpecies0,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(massSpecies1GPU,&massSpecies1,sizeof(double)));

  //Fluid density to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double)));

  double fact1, fact4, fact5;
  //FACT1 DIFFERENT FOR INCOMPRESSIBLE
  fact1 = sqrt((4.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  //FACT4 DIFFERENT FOR INCOMPRESSIBLE
  fact4 = sqrt((2.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  fact5 = sqrt(1./(dt*cVolume));
  cutilSafeCall(cudaMemcpyToSymbol(gradTemperatureGPU,&gradTemperature,sizeof(double)));
  
  //Prefactor for stochastic force to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact4GPU,&fact4,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact5GPU,&fact5,sizeof(double)));

  //Cell size to constant memory
  fact1 = lx/double(mx);
  double fact2 = ly/double(my);
  double fact3 = lz/double(mz);
  cutilSafeCall(cudaMemcpyToSymbol(dxGPU,&fact1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dyGPU,&fact2,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dzGPU,&fact3,sizeof(double)));

  //Inverse cell size to cosntant memory
  fact1 = double(mx)/lx;
  fact2 = double(my)/ly;
  fact3 = double(mz)/lz;
  cutilSafeCall(cudaMemcpyToSymbol(invdxGPU,&fact1,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invdyGPU,&fact2,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invdzGPU,&fact3,sizeof(double)));  

  //Inverse time step to constant memory
  fact1 = 1./dt;
  cutilSafeCall(cudaMemcpyToSymbol(invdtGPU,&fact1,sizeof(double)));

  //Inverse box size to constant memory
  fact1 = 1./lx;
  fact2 = 1./ly;
  fact3 = 1./lz;
  cutilSafeCall(cudaMemcpyToSymbol(invlxGPU,&fact1,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invlyGPU,&fact2,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invlzGPU,&fact3,sizeof(double)));

  //Some options to constant memory
  bool auxbool = 0;
  cutilSafeCall(cudaMemcpyToSymbol(setparticlesGPU,&auxbool,sizeof(bool)));
  cutilSafeCall(cudaMemcpyToSymbol(setboundaryGPU,&auxbool,sizeof(bool)));
  








  //Step to global memory
  long long auxulonglong = 0;
  cutilSafeCall(cudaMalloc((void**)&stepGPU,sizeof(long long)));
  cutilSafeCall(cudaMemcpy(stepGPU,&auxulonglong,sizeof(long long),cudaMemcpyHostToDevice));

  //Fluid velocity and velocity prediction to
  //global memory
  cutilSafeCall(cudaMalloc((void**)&vxGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzPredictionGPU,ncells*sizeof(double)));

  //Concentration to global memory
  cutilSafeCall(cudaMalloc((void**)&cGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&cPredictionGPU,ncells*sizeof(double)));

  //Centers cells to global memory
  cutilSafeCall(cudaMalloc((void**)&rxcellGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rycellGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzcellGPU,ncells*sizeof(double)));

  //List of neighbors cells to global memory
  cutilSafeCall(cudaMalloc((void**)&vecino0GPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino1GPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino2GPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino3GPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino4GPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino5GPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecinopxpyGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmyGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxpzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpyGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxmyGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxmzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopypzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopymzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomypzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomymzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxpypzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxpymzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmypzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmymzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpypzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpymzGPU,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecinomxmypzGPU,ncells*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxmymzGPU,ncells*sizeof(int))); 

  //Factors for the update in fourier space to global memory
  cutilSafeCall(cudaMalloc((void**)&gradKx,     mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKy,     my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKz,     mz*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKx,      mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKy,      my*sizeof(cufftDoubleComplex)));  
  cutilSafeCall(cudaMalloc((void**)&expKz,      mz*sizeof(cufftDoubleComplex)));

  cutilSafeCall(cudaMalloc((void**)&pF,sizeof(prefactorsFourier)));

  //Complex velocity field to global memory
  cutilSafeCall(cudaMalloc((void**)&vxZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vyZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vzZ,ncells*sizeof(cufftDoubleComplex))); 

  //Complex concentration to global memory
  cutilSafeCall(cudaMalloc((void**)&cZ,ncells*sizeof(cufftDoubleComplex)));


  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
