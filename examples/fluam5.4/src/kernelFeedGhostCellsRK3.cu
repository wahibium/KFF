// Filename: kernelFeedGhostCellsRK3.cu
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


__global__ void kernelFeedGhostCellsRK3(int* ghostToPIGPU, 
					int* ghostToGhostGPU,
					double* densityGPU, 
					double* densityPredictionGPU,
					double* vxGPU, 
					double* vyGPU, 
					double* vzGPU,
					double* vxPredictionGPU, 
					double* vyPredictionGPU, 
					double* vzPredictionGPU){

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
    

}
