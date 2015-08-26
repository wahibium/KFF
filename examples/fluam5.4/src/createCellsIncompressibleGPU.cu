// Filename: createCellsIncompressibleGPU.cu
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


bool createCellsIncompressibleGPU(){
  cutilSafeCall(cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(myGPU,&my,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int)));

  cutilSafeCall(cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int)));



  cutilSafeCall(cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool)));

  cutilSafeCall(cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double)));

  cutilSafeCall(cudaMalloc((void**)&vxGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzPredictionGPU,ncells*sizeof(double)));

 
  cutilSafeCall(cudaMalloc((void**)&rxcellGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rycellGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzcellGPU,ncells*sizeof(double)));

  //FACT1 DIFFERENT FOR INCOMPRESSIBLE
  double fact1 = sqrt((4.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  //FACT4 DIFFERENT FOR INCOMPRESSIBLE
  double fact4 = sqrt((2.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  double fact5 = sqrt(1./(dt*cVolume));

  cutilSafeCall(cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact4GPU,&fact4,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact5GPU,&fact5,sizeof(double)));


  fact1 = lx/double(mx);
  double fact2 = ly/double(my);
  double fact3 = lz/double(mz);
  cutilSafeCall(cudaMemcpyToSymbol(dxGPU,&fact1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dyGPU,&fact2,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dzGPU,&fact3,sizeof(double)));

  fact1 = double(mx)/lx;
  fact2 = double(my)/ly;
  fact3 = double(mz)/lz;
  cutilSafeCall(cudaMemcpyToSymbol(invdxGPU,&fact1,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invdyGPU,&fact2,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invdzGPU,&fact3,sizeof(double)));  
  fact1 = 1./dt;
  cutilSafeCall(cudaMemcpyToSymbol(invdtGPU,&fact1,sizeof(double)));
  fact1 = 1./lx;
  fact2 = 1./ly;
  fact3 = 1./lz;
  cutilSafeCall(cudaMemcpyToSymbol(invlxGPU,&fact1,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invlyGPU,&fact2,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invlzGPU,&fact3,sizeof(double)));

 
  bool auxbool = 0;
  cutilSafeCall(cudaMemcpyToSymbol(setparticlesGPU,&auxbool,sizeof(bool)));
  cutilSafeCall(cudaMemcpyToSymbol(setboundaryGPU,&auxbool,sizeof(bool)));


  long long auxulonglong = 0;
  cutilSafeCall(cudaMalloc((void**)&stepGPU,sizeof(long long)));
  cutilSafeCall(cudaMemcpy(stepGPU,&auxulonglong,sizeof(long long),cudaMemcpyHostToDevice));


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


  //Factors for the update in fourier space
  cutilSafeCall(cudaMalloc((void**)&gradKx,     mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKy,     my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKz,     mz*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKx,      mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKy,      my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKz,      mz*sizeof(cufftDoubleComplex)));

  cutilSafeCall(cudaMalloc((void**)&pF,sizeof(prefactorsFourier)));

  //cutilSafeCall(cudaMalloc((void**)&WxZ,ncells*sizeof(cufftDoubleComplex)));
  //cutilSafeCall(cudaMalloc((void**)&WyZ,ncells*sizeof(cufftDoubleComplex)));
  //cutilSafeCall(cudaMalloc((void**)&WzZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vxZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vyZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vzZ,ncells*sizeof(cufftDoubleComplex))); 

  if(quasiNeutrallyBuoyant || quasiNeutrallyBuoyant2D || quasiNeutrallyBuoyant4pt2D){
    cutilSafeCall(cudaMalloc((void**)&advXGPU,ncells*sizeof(double)));
    cutilSafeCall(cudaMalloc((void**)&advYGPU,ncells*sizeof(double)));
    cutilSafeCall(cudaMalloc((void**)&advZGPU,ncells*sizeof(double)));
  }

  
  cutilSafeCall(cudaMemcpyToSymbol(pressurea0GPU,&pressurea0,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(pressurea1GPU,&pressurea1,sizeof(double)));
  //cutilSafeCall(cudaMemcpyToSymbol(pressurea2GPU,&pressurea2,sizeof(double)));


  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
