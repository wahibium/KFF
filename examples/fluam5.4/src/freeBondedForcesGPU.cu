// Filename: freeBondedForcesGPU.cu
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


bool freeBondedForcesGPU(){

  
  cutilSafeCall(cudaFree(bondsParticleParticleGPU));
  cutilSafeCall(cudaFree(bondsParticleParticleOffsetGPU));
  cutilSafeCall(cudaFree(bondsIndexParticleParticleGPU));
  cutilSafeCall(cudaFree(r0ParticleParticleGPU));
  cutilSafeCall(cudaFree(kSpringParticleParticleGPU));


  cutilSafeCall(cudaFree(bondsParticleFixedPointGPU));
  cutilSafeCall(cudaFree(bondsParticleFixedPointOffsetGPU));
  //cutilSafeCall(cudaFree(bondsIndexParticleFixedPointGPU));
  cutilSafeCall(cudaFree(r0ParticleFixedPointGPU));
  cutilSafeCall(cudaFree(kSpringParticleFixedPointGPU));
  cutilSafeCall(cudaFree(rxFixedPointGPU));
  cutilSafeCall(cudaFree(ryFixedPointGPU));
  cutilSafeCall(cudaFree(rzFixedPointGPU));

  cutilSafeCall(cudaFree(bFV));
    
  return 1;
}


