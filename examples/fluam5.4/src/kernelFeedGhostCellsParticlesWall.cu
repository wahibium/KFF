// Filename: kernelFeedGhostCellsParticlesWall.cu
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


__global__ void kernelFeedGhostCellsParticlesWall(int* ghostToPIGPU, 
                				  int* ghostToGhostGPU,
						  double* densityGPU, 
						  double* densityPredictionGPU,
						  double* vxGPU, 
						  double* vyGPU, 
						  double* vzGPU,
						  double* vxPredictionGPU, 
						  double* vyPredictionGPU, 
						  double* vzPredictionGPU,
						  double* d_rand){

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
    



    //int fy = (ighost % (mxmytGPU)) / mxtGPU;    
    //if((fy!=0)&&(fy!=(mytGPU-1))){
      d_rand[ighost] = d_rand[ireal];
      d_rand[ighost+ncellstGPU] = d_rand[ireal+ncellstGPU];
      d_rand[ighost+2*ncellstGPU] = d_rand[ireal+2*ncellstGPU];
      d_rand[ighost+3*ncellstGPU] = d_rand[ireal+3*ncellstGPU];
      d_rand[ighost+4*ncellstGPU] = d_rand[ireal+4*ncellstGPU];
      d_rand[ighost+5*ncellstGPU] = d_rand[ireal+5*ncellstGPU];
      d_rand[ighost+6*ncellstGPU] = d_rand[ireal+6*ncellstGPU];
      d_rand[ighost+7*ncellstGPU] = d_rand[ireal+7*ncellstGPU];
      d_rand[ighost+8*ncellstGPU] = d_rand[ireal+8*ncellstGPU];
      d_rand[ighost+9*ncellstGPU] = d_rand[ireal+9*ncellstGPU];
      d_rand[ighost+10*ncellstGPU] = d_rand[ireal+10*ncellstGPU];
      d_rand[ighost+11*ncellstGPU] = d_rand[ireal+11*ncellstGPU];
      //}
      	

}



__global__ void kernelFeedGhostCellsParticlesWall2(int* ghostToPIGPU, 
						   int* ghostToGhostGPU,
						   double* densityGPU, 
						   double* densityPredictionGPU,
						   double* vxGPU, 
						   double* vyGPU, 
						   double* vzGPU,
						   double* vxPredictionGPU, 
						   double* vyPredictionGPU, 
						   double* vzPredictionGPU,
						   double* d_rand){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i>=(ncellstGPU-ncellsGPU)) return;
    int ighost = ghostToGhostGPU[i];
    int fy = (ighost % (mxmytGPU)) / mxtGPU;    
    


    if(fy==0){
      densityGPU[ighost] = densityGPU[ighost+mxtGPU];
      densityPredictionGPU[ighost] = densityPredictionGPU[ighost+mxtGPU];


      vxGPU[ighost] = 2*vxWall0GPU - vxGPU[ighost+mxtGPU];
      vxPredictionGPU[ighost] = 2*vxWall0GPU - vxPredictionGPU[ighost+mxtGPU];

      vyGPU[ighost] = 0;
      vyPredictionGPU[ighost] = 0;

      vzGPU[ighost] = 2*vzWall0GPU - vzGPU[ighost+mxtGPU];
      vzPredictionGPU[ighost] = 2*vzWall0GPU - vzPredictionGPU[ighost+mxtGPU];



      d_rand[ighost+ncellstGPU] = 1.4142135624 * d_rand[ighost];
      d_rand[ighost+4*ncellstGPU] = 1.4142135624 * d_rand[ighost+3*ncellstGPU];
      d_rand[ighost+7*ncellstGPU] = 1.4142135624 * d_rand[ighost+6*ncellstGPU];
      d_rand[ighost+10*ncellstGPU] = 1.4142135624 * d_rand[ighost+9*ncellstGPU];
            
    }
    else if(fy==(mytGPU-1)){
      densityGPU[ighost] = densityGPU[ighost-mxtGPU];
      densityPredictionGPU[ighost] = densityPredictionGPU[ighost-mxtGPU];


      vxGPU[ighost] = 2*vxWall1GPU - vxGPU[ighost-mxtGPU];
      vxPredictionGPU[ighost] = 2*vxWall1GPU - vxPredictionGPU[ighost-mxtGPU];


      vyGPU[ighost-mxtGPU] = 0;
      vyPredictionGPU[ighost-mxtGPU] = 0;
      vyGPU[ighost] = 0;
      vyPredictionGPU[ighost] = 0;

      vzGPU[ighost] = 2*vzWall1GPU - vzGPU[ighost-mxtGPU];
      vzPredictionGPU[ighost] = 2*vzWall1GPU - vzPredictionGPU[ighost-mxtGPU];


      d_rand[ighost-mxtGPU+ncellstGPU] = 1.4142135624 * d_rand[ighost+ncellstGPU];
      d_rand[ighost-mxtGPU+4*ncellstGPU] = 1.4142135624 * d_rand[ighost+4*ncellstGPU];
      d_rand[ighost-mxtGPU+7*ncellstGPU] = 1.4142135624 * d_rand[ighost+7*ncellstGPU];
      d_rand[ighost-mxtGPU+10*ncellstGPU] = 1.4142135624 * d_rand[ighost+10*ncellstGPU];
      

    }
    


}
