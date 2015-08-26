// Filename: headerGPU.h
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
				  int* neighbormxmymzGPU);



__global__ void initializeVecinos2(int* neighbor0GPU, 
				   int* neighbor1GPU, 
				   int* neighbor2GPU,
				   int* neighbor3GPU, 
				   int* neighbor4GPU, 
				   int*neighbor5GPU);
