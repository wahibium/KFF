// Filename: kernelFeedGhostCellsBinaryMixture.cu
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


__global__ void kernelFeedGhostCellsBinaryMixture(int* ghostToPIGPU, 
						  int* ghostToGhostGPU,
						  double* densityGPU, 
						  double* densityPredictionGPU,
						  double* vxGPU, 
						  double* vyGPU, 
						  double* vzGPU,
						  double* vxPredictionGPU, 
						  double* vyPredictionGPU, 
						  double* vzPredictionGPU,
						  double* cGPU, 
						  double* cPredictionGPU, 
						  double* d_rand, 
						  int substep){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i>=(ncellstGPU-ncellsGPU)) return;
    int ireal = ghostToPIGPU[i];
    int ighost = ghostToGhostGPU[i];
    densityGPU[ighost] = densityGPU[ireal];
    densityPredictionGPU[ighost] = densityPredictionGPU[ireal];
    vxGPU[ighost] = vxGPU[ireal];
    vyGPU[ighost] = vyGPU[ireal];
    vzGPU[ighost] = vzGPU[ireal];
    vxPredictionGPU[ighost] = vxPredictionGPU[ireal];
    vyPredictionGPU[ighost] = vyPredictionGPU[ireal];
    vzPredictionGPU[ighost] = vzPredictionGPU[ireal];

    cGPU[ighost] = cGPU[ireal];
    cPredictionGPU[ighost] = cPredictionGPU[ireal];

    
    int n0;
    n0 = substep * ncellstGPU * 18;
    d_rand[n0+ighost] = d_rand[n0+ireal];
    d_rand[n0+ighost+ncellstGPU] = d_rand[n0+ireal+ncellstGPU];
    d_rand[n0+ighost+2*ncellstGPU] = d_rand[n0+ireal+2*ncellstGPU];
    d_rand[n0+ighost+3*ncellstGPU] = d_rand[n0+ireal+3*ncellstGPU];
    d_rand[n0+ighost+4*ncellstGPU] = d_rand[n0+ireal+4*ncellstGPU];
    d_rand[n0+ighost+5*ncellstGPU] = d_rand[n0+ireal+5*ncellstGPU];
    d_rand[n0+ighost+6*ncellstGPU] = d_rand[n0+ireal+6*ncellstGPU];
    d_rand[n0+ighost+7*ncellstGPU] = d_rand[n0+ireal+7*ncellstGPU];
    d_rand[n0+ighost+8*ncellstGPU] = d_rand[n0+ireal+8*ncellstGPU];
    d_rand[n0+ighost+9*ncellstGPU] = d_rand[n0+ireal+9*ncellstGPU];
    d_rand[n0+ighost+10*ncellstGPU] = d_rand[n0+ireal+10*ncellstGPU];
    d_rand[n0+ighost+11*ncellstGPU] = d_rand[n0+ireal+11*ncellstGPU];
    d_rand[n0+ighost+12*ncellstGPU] = d_rand[n0+ireal+12*ncellstGPU];
    d_rand[n0+ighost+13*ncellstGPU] = d_rand[n0+ireal+13*ncellstGPU];
    d_rand[n0+ighost+14*ncellstGPU] = d_rand[n0+ireal+14*ncellstGPU];
    d_rand[n0+ighost+15*ncellstGPU] = d_rand[n0+ireal+15*ncellstGPU];
    d_rand[n0+ighost+16*ncellstGPU] = d_rand[n0+ireal+16*ncellstGPU];
    d_rand[n0+ighost+17*ncellstGPU] = d_rand[n0+ireal+17*ncellstGPU];    

}
