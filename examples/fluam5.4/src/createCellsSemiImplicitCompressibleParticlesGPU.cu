// Filename: createCellsGPU.cu
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


bool createCellsSemiImplicitCompressibleParticlesGPU(){



  cutilSafeCall(cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(myGPU,&my,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int)));

  cutilSafeCall(cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int)));

  int aux = (mxt) * (myt);
  cutilSafeCall(cudaMemcpyToSymbol(mxmytGPU,&aux,sizeof(int)));


  cutilSafeCall(cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(bulkviscosityGPU,&bulkviscosity,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(pressurea0GPU,&pressurea0,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(pressurea1GPU,&pressurea1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(pressurea2GPU,&pressurea2,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool)));

  cutilSafeCall(cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double)));



  cutilSafeCall(cudaMalloc((void**)&densityGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzGPU,ncellst*sizeof(double)));

  cutilSafeCall(cudaMalloc((void**)&vxPredictionGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyPredictionGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzPredictionGPU,ncellst*sizeof(double)));

 

  cutilSafeCall(cudaMalloc((void**)&dpxGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&dpyGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&dpzGPU,ncellst*sizeof(double)));

  cutilSafeCall(cudaMalloc((void**)&rxcellGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rycellGPU,ncellst*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzcellGPU,ncellst*sizeof(double)));

  double fact1 = sqrt((4.*temperature*shearviscosity)/(dt*cVolume));
  double fact2 = sqrt((2.*temperature*bulkviscosity)/(3.*dt*cVolume));
  double fact3 = bulkviscosity - 2. * shearviscosity/3.;
  double fact4 = sqrt((2.*temperature*shearviscosity)/(dt*cVolume));
  double fact5 = sqrt(1./(dt*cVolume));

  cutilSafeCall(cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact2GPU,&fact2,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact3GPU,&fact3,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact4GPU,&fact4,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(fact5GPU,&fact5,sizeof(double)));


  fact1 = lx/double(mx);
  fact2 = ly/double(my);
  fact3 = lz/double(mz);
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


  cutilSafeCall(cudaMalloc((void**)&vecino0GPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino1GPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino2GPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino3GPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino4GPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecino5GPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecinopxpyGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmyGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxpzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpyGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxmyGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxmzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopypzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopymzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomypzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomymzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxpypzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxpymzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmypzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinopxmymzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpypzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxpymzGPU,ncellst*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&vecinomxmypzGPU,ncellst*sizeof(int))); 
  cutilSafeCall(cudaMalloc((void**)&vecinomxmymzGPU,ncellst*sizeof(int))); 






  //Factors for the update in fourier space
  cutilSafeCall(cudaMalloc((void**)&gradKx,     mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKy,     my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKz,     mz*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKx,      mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKy,      my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKz,      mz*sizeof(cufftDoubleComplex)));

  cutilSafeCall(cudaMalloc((void**)&pF,sizeof(prefactorsFourier)));

  cutilSafeCall(cudaMalloc((void**)&vxZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vyZ,ncells*sizeof(cufftDoubleComplex)));




  double auxD[ncellst];
  for(int i=0;i<ncellst;i++) auxD[i] = 0;
  cudaMemcpy(dpxGPU,auxD,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dpyGPU,auxD,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dpzGPU,auxD,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vxPredictionGPU,auxD,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vyPredictionGPU,auxD,ncellst*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vzPredictionGPU,auxD,ncellst*sizeof(double),cudaMemcpyHostToDevice);

  cufftDoubleComplex auxC[ncellst];
  for(int i=0;i<ncellst;i++){
    auxC[i].x = 1;
    auxC[i].y = 0;
  }
  cudaMemcpy(vxZ,auxC,ncellst*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice);
  cudaMemcpy(vyZ,auxC,ncellst*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice);


  
  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}




