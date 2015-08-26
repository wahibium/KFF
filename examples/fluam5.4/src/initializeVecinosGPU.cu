// Filename: initializeVecinosGPU.cu
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


__global__ void initializeVecinos(int* neighbor1GPU, 
				  int* neighbor2GPU, 
				  int* neighbor3GPU, 
				  int* neighbor4GPU, 
				  int* neighborpxpyGPU, 
				  int* neighborpxmyGPU, 
				  int* neighborpxpzGPU, 
				  int* neighborpxmzGPU,
				  int* neighbormxpyGPU, 
				  int* neighbormxmyGPU, 
				  int* neighbormxpzGPU, 
				  int* neighbormxmzGPU,
				  int* neighborpypzGPU, 
				  int* neighborpymzGPU, 
				  int* neighbormypzGPU, 
				  int* neighbormymzGPU,
				  int* neighborpxpypzGPU, 
				  int* neighborpxpymzGPU, 
				  int* neighborpxmypzGPU, 
				  int* neighborpxmymzGPU,
				  int* neighbormxpypzGPU, 
				  int* neighbormxpymzGPU, 
				  int* neighbormxmypzGPU, 
				  int* neighbormxmymzGPU){

  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellstGPU) return;

  int mxmy = mxGPU * mytGPU; 
  int neighbor1, neighbor2, neighbor3, neighbor4;
  {
    int fz = i/mxmy;
    int fy = (i % mxmy)/mxGPU;
    int fx = i%mxGPU;
     
    int fyp1 = ((fy+1) % mytGPU);
    int fym1 = ((fy-1+mytGPU) % mytGPU);
    int fxp1 = ((fx+1) % mxGPU);
    int fxm1 = ((fx-1+mxGPU) % mxGPU);
    
    
    neighbor1 = fz * mxmy   +fym1 * mxGPU + fx;
    neighbor2 = fz * mxmy   +fy   * mxGPU + fxm1;
    neighbor3 = fz * mxmy   +fy   * mxGPU + fxp1;
    neighbor4 = fz * mxmy   +fyp1 * mxGPU + fx;
    
  }

  int neighborpxpy, neighborpxmy, neighborpxpz, neighborpxmz;
  int neighbormxpy, neighbormxmy, neighbormxpz, neighbormxmz;
  int neighborpypz, neighborpymz, neighbormypz, neighbormymz;

  int neighborpxpypz, neighborpxpymz, neighborpxmypz, neighborpxmymz;
  int neighbormxpypz, neighbormxpymz, neighbormxmypz, neighbormxmymz;
  
  {
    //NEIGHBOUR NEIGHBOR3
    int fz = neighbor3/mxmy;
    int fy = (neighbor3 % mxmy)/mxGPU;
    int fx = neighbor3 % mxGPU;
    int fzp1, fzm1, fyp1, fym1;
    
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 
    fyp1 = ((fy+1) % mytGPU) * mxGPU;
    fym1 = ((fy-1+mytGPU) % mytGPU) * mxGPU;

    fz = fz * mxmy;
    fy = fy * mxGPU;
    neighborpxmz = fzm1 + fy   + fx;
    neighborpxmy = fz   + fym1 + fx;

    neighborpxpy = fz   + fyp1 + fx;
    neighborpxpz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBOR2
    fz = neighbor2/mxmy;
    fy = (neighbor2 % mxmy)/mxGPU;
    fx = neighbor2 % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 
    fyp1 = ((fy+1) % mytGPU) * mxGPU;
    fym1 = ((fy-1+mytGPU) % mytGPU) * mxGPU;

    fz = fz * mxmy;
    fy = fy * mxGPU;
    neighbormxmz = fzm1 + fy   + fx;
    neighbormxmy = fz   + fym1 + fx;

    neighbormxpy = fz   + fyp1 + fx;
    neighbormxpz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBOR4
    fz = neighbor4/mxmy;
    fy = (neighbor4 % mxmy)/mxGPU;
    fx = neighbor4 % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 

    fy = fy * mxGPU;
    neighborpymz = fzm1 + fy   + fx;

    neighborpypz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBOR1
    fz = neighbor1/mxmy;
    fy = (neighbor1 % mxmy)/mxGPU;
    fx = neighbor1 % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 

    fy = fy * mxGPU;
    neighbormymz = fzm1 + fy   + fx;

    neighbormypz = fzp1 + fy   + fx;    
    
    //NEIGHBOUR NEIGHBORPXPY
    fz = neighborpxpy/mxmy;
    fy = (neighborpxpy % mxmy)/mxGPU;
    fx = neighborpxpy % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 

    fy = fy * mxGPU;
    neighborpxpymz = fzm1 + fy   + fx;

    neighborpxpypz = fzp1 + fy   + fx;    
        
    //NEIGHBOUR NEIGHBORPXMY
    fz = neighborpxmy/mxmy;
    fy = (neighborpxmy % mxmy)/mxGPU;
    fx = neighborpxmy % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 

    fy = fy * mxGPU;
    neighborpxmymz = fzm1 + fy   + fx;

    neighborpxmypz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBORMXPY
    fz = neighbormxpy/mxmy;
    fy = (neighbormxpy % mxmy)/mxGPU;
    fx = neighbormxpy % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 

    fy = fy * mxGPU;
    neighbormxpymz = fzm1 + fy   + fx;

    neighbormxpypz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBORMXMY
    fz = neighbormxmy/mxmy;
    fy = (neighbormxmy % mxmy)/mxGPU;
    fx = neighbormxmy % mxGPU;
    fzp1 = ((fz+1) % mzGPU) * mxmy;
    fzm1 = ((fz-1+mzGPU) % mzGPU) * mxmy; 

    fy = fy * mxGPU;
    neighbormxmymz = fzm1 + fy   + fx;

    neighbormxmypz = fzp1 + fy   + fx;    
  }  
  
  neighborpxpyGPU[i] = neighborpxpy;
  neighborpxmyGPU[i] = neighborpxmy;
  neighborpxpzGPU[i] = neighborpxpz;
  neighborpxmzGPU[i] = neighborpxmz;
  neighbormxpyGPU[i] = neighbormxpy;
  neighbormxmyGPU[i] = neighbormxmy;
  neighbormxpzGPU[i] = neighbormxpz;
  neighbormxmzGPU[i] = neighbormxmz;
  neighborpypzGPU[i] = neighborpypz;
  neighborpymzGPU[i] = neighborpymz;
  neighbormypzGPU[i] = neighbormypz;
  neighbormymzGPU[i] = neighbormymz;
  neighborpxpypzGPU[i] = neighborpxpypz;
  neighborpxpymzGPU[i] = neighborpxpymz;
  neighborpxmypzGPU[i] = neighborpxmypz;
  neighborpxmymzGPU[i] = neighborpxmymz;
  neighbormxpypzGPU[i] = neighbormxpypz;
  neighbormxpymzGPU[i] = neighbormxpymz;
  neighbormxmypzGPU[i] = neighbormxmypz;
  neighbormxmymzGPU[i] = neighbormxmymz;
}


__global__ void initializeVecinos2(int* neighbor0GPU, 
				   int* neighbor1GPU, 
				   int* neighbor2GPU,
				   int* neighbor3GPU, 
				   int* neighbor4GPU, 
				   int* neighbor5GPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellstGPU) return;
  
  int mxmy = mxGPU * mytGPU; 
  int neighbor0, neighbor1, neighbor2, neighbor3, neighbor4, neighbor5;
  {
    int fz = i/(mxmy);
    int fy = (i % mxmy)/mxGPU;
    int fx = i%mxGPU;
    int fzp1 = ((fz+1) % mzGPU);
    int fzm1 = ((fz-1+mzGPU) % mzGPU); 
    int fyp1 = ((fy+1) % mytGPU);
    int fym1 = ((fy-1+mytGPU) % mytGPU);
    int fxp1 = ((fx+1) % mxGPU);
    int fxm1 = ((fx-1+mxGPU) % mxGPU);
    

    neighbor0 = fzm1 * mxmy + fy   * mxGPU + fx;
    neighbor1 = fz   * mxmy + fym1 * mxGPU + fx;
    neighbor2 = fz   * mxmy + fy   * mxGPU + fxm1;
    neighbor3 = fz   * mxmy + fy   * mxGPU + fxp1;
    neighbor4 = fz   * mxmy + fyp1 * mxGPU + fx;
    neighbor5 = fzp1 * mxmy + fy   * mxGPU + fx;

  }

  neighbor0GPU[i] = neighbor0;
  neighbor1GPU[i] = neighbor1;
  neighbor2GPU[i] = neighbor2;
  neighbor3GPU[i] = neighbor3;
  neighbor4GPU[i] = neighbor4;
  neighbor5GPU[i] = neighbor5;  

}
