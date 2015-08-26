// Filename: initializeNeighborsGPU.cu
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


__global__ void initializeNeighbors(int* neighbor1GPU, 
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
  if(i>=mNeighborsGPU) return;

  int mxmy = mxNeighborsGPU * myNeighborsGPU; 
  int neighbor1, neighbor2, neighbor3, neighbor4;
  {
    int fz = i/(mxNeighborsGPU*myNeighborsGPU);
    int fy = (i % (mxNeighborsGPU*myNeighborsGPU))/mxNeighborsGPU;
    int fx = i%mxNeighborsGPU;
     
    int fyp1 = ((fy+1) % myNeighborsGPU);
    int fym1 = ((fy-1+myNeighborsGPU) % myNeighborsGPU);
    int fxp1 = ((fx+1) % mxNeighborsGPU);
    int fxm1 = ((fx-1+mxNeighborsGPU) % mxNeighborsGPU);
    
    
    neighbor1 = fz * mxNeighborsGPU*myNeighborsGPU   +fym1 * mxNeighborsGPU + fx;
    neighbor2 = fz * mxNeighborsGPU*myNeighborsGPU   +fy   * mxNeighborsGPU + fxm1;
    neighbor3 = fz * mxNeighborsGPU*myNeighborsGPU   +fy   * mxNeighborsGPU + fxp1;
    neighbor4 = fz * mxNeighborsGPU*myNeighborsGPU   +fyp1 * mxNeighborsGPU + fx;
    
  }

  //neighbor0GPU[i] = neighbor0;
  //neighbor1GPU[i] = neighbor1;
  //neighbor2GPU[i] = neighbor2;
  //neighbor3GPU[i] = neighbor3;
  //neighbor4GPU[i] = neighbor4;
  //neighbor5GPU[i] = neighbor5;  

  int neighborpxpy, neighborpxmy, neighborpxpz, neighborpxmz;
  int neighbormxpy, neighbormxmy, neighbormxpz, neighbormxmz;
  int neighborpypz, neighborpymz, neighbormypz, neighbormymz;

  int neighborpxpypz, neighborpxpymz, neighborpxmypz, neighborpxmymz;
  int neighbormxpypz, neighbormxpymz, neighbormxmypz, neighbormxmymz;
  
  {
    //NEIGHBOUR NEIGHBOR3
    int fz = neighbor3/mxmy;
    int fy = (neighbor3 % mxmy)/mxNeighborsGPU;
    int fx = neighbor3 % mxNeighborsGPU;
    int fzp1, fzm1, fyp1, fym1;
    
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 
    fyp1 = ((fy+1) % myNeighborsGPU) * mxNeighborsGPU;
    fym1 = ((fy-1+myNeighborsGPU) % myNeighborsGPU) * mxNeighborsGPU;

    fz = fz * mxmy;
    fy = fy * mxNeighborsGPU;
    neighborpxmz = fzm1 + fy   + fx;
    neighborpxmy = fz   + fym1 + fx;

    neighborpxpy = fz   + fyp1 + fx;
    neighborpxpz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBOR2
    fz = neighbor2/mxmy;
    fy = (neighbor2 % mxmy)/mxNeighborsGPU;
    fx = neighbor2 % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 
    fyp1 = ((fy+1) % myNeighborsGPU) * mxNeighborsGPU;
    fym1 = ((fy-1+myNeighborsGPU) % myNeighborsGPU) * mxNeighborsGPU;

    fz = fz * mxmy;
    fy = fy * mxNeighborsGPU;
    neighbormxmz = fzm1 + fy   + fx;
    neighbormxmy = fz   + fym1 + fx;

    neighbormxpy = fz   + fyp1 + fx;
    neighbormxpz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBOR4
    fz = neighbor4/mxmy;
    fy = (neighbor4 % mxmy)/mxNeighborsGPU;
    fx = neighbor4 % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 

    fy = fy * mxNeighborsGPU;
    neighborpymz = fzm1 + fy   + fx;

    neighborpypz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBOR1
    fz = neighbor1/mxmy;
    fy = (neighbor1 % mxmy)/mxNeighborsGPU;
    fx = neighbor1 % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 

    fy = fy * mxNeighborsGPU;
    neighbormymz = fzm1 + fy   + fx;

    neighbormypz = fzp1 + fy   + fx;    
    
    //NEIGHBOUR NEIGHBORPXPY
    fz = neighborpxpy/mxmy;
    fy = (neighborpxpy % mxmy)/mxNeighborsGPU;
    fx = neighborpxpy % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 

    fy = fy * mxNeighborsGPU;
    neighborpxpymz = fzm1 + fy   + fx;

    neighborpxpypz = fzp1 + fy   + fx;    
        
    //NEIGHBOUR NEIGHBORPXMY
    fz = neighborpxmy/mxmy;
    fy = (neighborpxmy % mxmy)/mxNeighborsGPU;
    fx = neighborpxmy % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 

    fy = fy * mxNeighborsGPU;
    neighborpxmymz = fzm1 + fy   + fx;

    neighborpxmypz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBORMXPY
    fz = neighbormxpy/mxmy;
    fy = (neighbormxpy % mxmy)/mxNeighborsGPU;
    fx = neighbormxpy % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 

    fy = fy * mxNeighborsGPU;
    neighbormxpymz = fzm1 + fy   + fx;

    neighbormxpypz = fzp1 + fy   + fx;    

    //NEIGHBOUR NEIGHBORMXMY
    fz = neighbormxmy/mxmy;
    fy = (neighbormxmy % mxmy)/mxNeighborsGPU;
    fx = neighbormxmy % mxNeighborsGPU;
    fzp1 = ((fz+1) % mzNeighborsGPU) * mxmy;
    fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU) * mxmy; 

    fy = fy * mxNeighborsGPU;
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


__global__ void initializeNeighbors2(int* neighbor0GPU, 
				     int* neighbor1GPU, 
				     int* neighbor2GPU,
				     int* neighbor3GPU, 
				     int* neighbor4GPU, 
				     int*neighbor5GPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=mNeighborsGPU) return;
  
  int mxmy = mxNeighborsGPU * myNeighborsGPU; 
  int neighbor0, neighbor1, neighbor2, neighbor3, neighbor4, neighbor5;
  {
    int fz = i/(mxNeighborsGPU*myNeighborsGPU);
    int fy = (i % (mxNeighborsGPU*myNeighborsGPU))/mxNeighborsGPU;
    int fx = i%mxNeighborsGPU;
    int fzp1 = ((fz+1) % mzNeighborsGPU);
    int fzm1 = ((fz-1+mzNeighborsGPU) % mzNeighborsGPU); 
    int fyp1 = ((fy+1) % myNeighborsGPU);
    int fym1 = ((fy-1+myNeighborsGPU) % myNeighborsGPU);
    int fxp1 = ((fx+1) % mxNeighborsGPU);
    int fxm1 = ((fx-1+mxNeighborsGPU) % mxNeighborsGPU);
    
    /*neighbor0 = (fzm1%mzNeighborsGPU)*mxNeighborsGPU*myNeighborsGPU +(fy%myNeighborsGPU)*mxNeighborsGPU   +(fx%mxNeighborsGPU);
    neighbor1 = (fz%mzNeighborsGPU)*mxNeighborsGPU*myNeighborsGPU   +(fym1%myNeighborsGPU)*mxNeighborsGPU +(fx%mxNeighborsGPU);
    neighbor2 = (fz%mzNeighborsGPU)*mxNeighborsGPU*myNeighborsGPU   +(fy%myNeighborsGPU)*mxNeighborsGPU   +(fxm1%mxNeighborsGPU);
    neighbor3 = (fz%mzNeighborsGPU)*mxNeighborsGPU*myNeighborsGPU   +(fy%myNeighborsGPU)*mxNeighborsGPU   +(fxp1%mxNeighborsGPU);
    neighbor4 = (fz%mzNeighborsGPU)*mxNeighborsGPU*myNeighborsGPU   +(fyp1%myNeighborsGPU)*mxNeighborsGPU +(fx%mxNeighborsGPU);
    neighbor5 = (fzp1%mzNeighborsGPU)*mxNeighborsGPU*myNeighborsGPU +(fyp1%myNeighborsGPU)*mxNeighborsGPU +(fx%mxNeighborsGPU);*/
    neighbor0 = fzm1 * mxmy + fy   * mxNeighborsGPU + fx;
    neighbor1 = fz   * mxmy + fym1 * mxNeighborsGPU + fx;
    neighbor2 = fz   * mxmy + fy   * mxNeighborsGPU + fxm1;
    neighbor3 = fz   * mxmy + fy   * mxNeighborsGPU + fxp1;
    neighbor4 = fz   * mxmy + fyp1 * mxNeighborsGPU + fx;
    neighbor5 = fzp1 * mxmy + fy   * mxNeighborsGPU + fx;
    //printf("i neighbor0 %i %i %i\n",i,neighbor4,fzm1);
  }

  neighbor0GPU[i] = neighbor0;
  neighbor1GPU[i] = neighbor1;
  neighbor2GPU[i] = neighbor2;
  neighbor3GPU[i] = neighbor3;
  neighbor4GPU[i] = neighbor4;
  neighbor5GPU[i] = neighbor5;  

}
