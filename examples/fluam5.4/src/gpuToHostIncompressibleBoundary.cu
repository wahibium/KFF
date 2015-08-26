// Filename: gpuToHostIncompressibleBoundary.cu
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


bool gpuToHostIncompressibleBoundary(){

  int auxb[5];
  cutilSafeCall(cudaMemcpy(auxb,errorKernel,5*sizeof(int),cudaMemcpyDeviceToHost));
  if(auxb[0] == 1){
    for(int i=0;i<5;i++){
      cout << "ERROR IN KERNEL " << i << " " << auxb[i] << endl;
    }
    return 0;
  }

  cutilSafeCall(cudaMemcpy(cvx,vxGPU,ncells*sizeof(double),cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(cvy,vyGPU,ncells*sizeof(double),cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(cvz,vzGPU,ncells*sizeof(double),cudaMemcpyDeviceToHost));

  if(setparticles){
    cutilSafeCall(cudaMemcpy(rxParticle,&rxboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(ryParticle,&ryboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(rzParticle,&rzboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(vxParticle,&vxboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(vyParticle,&vyboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(vzParticle,&vzboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
  }

  return 1;
}
