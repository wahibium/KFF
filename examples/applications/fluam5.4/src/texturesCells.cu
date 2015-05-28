// Filename: texturesCells.cu
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


bool texturesCells(){
  //texvecino0GPU
  texvecino0GPU.normalized = false;
  texvecino0GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino0GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino0GPU,vecino0GPU,ncellst*sizeof(int)));
  //texvecino1GPU
  texvecino1GPU.normalized = false;
  texvecino1GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino1GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino1GPU,vecino1GPU,ncellst*sizeof(int)));
  //texvecino2GPU
  texvecino2GPU.normalized = false;
  texvecino2GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino2GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino2GPU,vecino2GPU,ncellst*sizeof(int)));
  //texvecino3GPU
  texvecino3GPU.normalized = false;
  texvecino3GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino3GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino3GPU,vecino3GPU,ncellst*sizeof(int)));
  //texvecino4GPU
  texvecino4GPU.normalized = false;
  texvecino4GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino4GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino4GPU,vecino4GPU,ncellst*sizeof(int)));
  //texvecino5GPU
  texvecino5GPU.normalized = false;
  texvecino5GPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecino5GPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecino5GPU,vecino5GPU,ncellst*sizeof(int)));
  //texvecinopxpyGPU
  texvecinopxpyGPU.normalized = false;
  texvecinopxpyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpyGPU,vecinopxpyGPU,ncellst*sizeof(int)));
  //texvecinopxmyGPU
  texvecinopxmyGPU.normalized = false;
  texvecinopxmyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmyGPU,vecinopxmyGPU,ncellst*sizeof(int)));
  //texvecinopxpzGPU
  texvecinopxpzGPU.normalized = false;
  texvecinopxpzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpzGPU,vecinopxpzGPU,ncellst*sizeof(int)));
  //texvecinopxmzGPU
  texvecinopxmzGPU.normalized = false;
  texvecinopxmzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmzGPU,vecinopxmzGPU,ncellst*sizeof(int)));
  //texvecinomxpyGPU
  texvecinomxpyGPU.normalized = false;
  texvecinomxpyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpyGPU,vecinomxpyGPU,ncellst*sizeof(int)));
  //texvecinomxmyGPU
  texvecinomxmyGPU.normalized = false;
  texvecinomxmyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmyGPU,vecinomxmyGPU,ncellst*sizeof(int)));
  //texvecinomxpzGPU
  texvecinomxpzGPU.normalized = false;
  texvecinomxpzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpzGPU,vecinomxpzGPU,ncellst*sizeof(int)));
  //texvecinomxmzGPU
  texvecinomxmzGPU.normalized = false;
  texvecinomxmzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmzGPU,vecinomxmzGPU,ncellst*sizeof(int)));
  //texvecinopypzGPU
  texvecinopypzGPU.normalized = false;
  texvecinopypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopypzGPU,vecinopypzGPU,ncellst*sizeof(int)));
  //texvecinopymzGPU
  texvecinopymzGPU.normalized = false;
  texvecinopymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopymzGPU,vecinopymzGPU,ncellst*sizeof(int)));
  //texvecinomypzGPU
  texvecinomypzGPU.normalized = false;
  texvecinomypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomypzGPU,vecinomypzGPU,ncellst*sizeof(int)));
  //texvecinomymzGPU
  texvecinomymzGPU.normalized = false;
  texvecinomymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomymzGPU,vecinomymzGPU,ncellst*sizeof(int)));
  //texvecinopxpypzGPU
  texvecinopxpypzGPU.normalized = false;
  texvecinopxpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpypzGPU,vecinopxpypzGPU,ncellst*sizeof(int)));
  //texvecinopxpymzGPU
  texvecinopxpymzGPU.normalized = false;
  texvecinopxpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxpymzGPU,vecinopxpymzGPU,ncellst*sizeof(int)));
  //texvecinopxmypzGPU
  texvecinopxmypzGPU.normalized = false;
  texvecinopxmypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmypzGPU,vecinopxmypzGPU,ncellst*sizeof(int)));
  //texvecinopxmymzGPU
  texvecinopxmymzGPU.normalized = false;
  texvecinopxmymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinopxmymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinopxmymzGPU,vecinopxmymzGPU,ncellst*sizeof(int)));
  //texvecinomxpypzGPU
  texvecinomxpypzGPU.normalized = false;
  texvecinomxpypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpypzGPU,vecinomxpypzGPU,ncellst*sizeof(int)));
  //texvecinomymzGPU
  texvecinomxpymzGPU.normalized = false;
  texvecinomxpymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxpymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxpymzGPU,vecinomxpymzGPU,ncellst*sizeof(int)));
  //texvecinomxmypzGPU
  texvecinomxmypzGPU.normalized = false;
  texvecinomxmypzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmypzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmypzGPU,vecinomxmypzGPU,ncellst*sizeof(int)));
  //texvecinomxmymzGPU
  texvecinomxmymzGPU.normalized = false;
  texvecinomxmymzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texvecinomxmymzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texvecinomxmymzGPU,vecinomxmymzGPU,ncellst*sizeof(int)));



  //texVxGPU;
  texVxGPU.normalized = false;
  texVxGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texVxGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncellst*sizeof(double)));
  //texVyGPU;
  texVyGPU.normalized = false;
  texVyGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texVyGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncellst*sizeof(double)));
  //texVzGPU;
  texVzGPU.normalized = false;
  texVzGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texVzGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncellst*sizeof(double)));



  cout << "TEXTURES CELLS :                DONE " << endl;

  return 1;

}
