// Filename: texturesCellsGhost.cu
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


bool texturesCellsGhost(){

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
