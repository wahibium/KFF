// Filename: realToComplex.cu
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


__global__ void doubleToDoubleComplex(double* vxGPU, 
				      double* vyGPU, 
				      double* vzGPU,
				      cufftDoubleComplex* vxI,
				      cufftDoubleComplex* vyI, 
				      cufftDoubleComplex* vzI){
  
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   
  
  vxI[j].x = vxGPU[j];
  vxI[j].y = 0;
  vyI[j].x = vyGPU[j];
  vyI[j].y = 0;
  vzI[j].x = vzGPU[j];
  vzI[j].y = 0;
}


__global__ void doubleComplexToDoubleNormalized(cufftDoubleComplex* vxUpI, 
						cufftDoubleComplex* vyUpI,
						cufftDoubleComplex* vzUpI, 
						double* vxGPU, 
						double* vyGPU, 
						double* vzGPU){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   
  
  vxGPU[j] = vxUpI[j].x / double(ncellsGPU);
  vyGPU[j] = vyUpI[j].x / double(ncellsGPU);
  vzGPU[j] = vzUpI[j].x / double(ncellsGPU);

}













__global__ void doubleComplexToDoubleNormalizedBinaryMixture(cufftDoubleComplex* vxUpI, 
							     cufftDoubleComplex* vyUpI,
							     cufftDoubleComplex* vzUpI,
							     cufftDoubleComplex* cZ, 
							     double* vxGPU, 
							     double* vyGPU, 
							     double* vzGPU,
							     double* cGPU){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   
  
  vxGPU[j] = vxUpI[j].x / double(ncellsGPU);
  vyGPU[j] = vxUpI[j].y / double(ncellsGPU);
  vzGPU[j] = vzUpI[j].x / double(ncellsGPU);
  cGPU[j]  = vzUpI[j].y / double(ncellsGPU);

  //vxGPU[j] = vxUpI[j].x / double(ncellsGPU);
  //vyGPU[j] = vyUpI[j].x / double(ncellsGPU);
  //vzGPU[j] = vzUpI[j].x / double(ncellsGPU);
  //cGPU[j] = cZ[j].x / double(ncellsGPU);
  
}













__global__ void calculateVelocityPrediction(cufftDoubleComplex* vxUpI, 
					    cufftDoubleComplex* vyUpI,
					    cufftDoubleComplex* vzUpI, 
					    double* vxGPU, 
					    double* vyGPU, 
					    double* vzGPU){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   

  vxGPU[j] = 0.5 * ( (vxUpI[j].x / double(ncellsGPU)) + fetch_double(texVxGPU,j));
  vyGPU[j] = 0.5 * ( (vyUpI[j].x / double(ncellsGPU)) + fetch_double(texVyGPU,j));
  vzGPU[j] = 0.5 * ( (vzUpI[j].x / double(ncellsGPU)) + fetch_double(texVzGPU,j));
  
}



















__global__ void calculateVelocityPredictionBinaryMixture(cufftDoubleComplex* vxUpI, 
							 cufftDoubleComplex* vyUpI,
							 cufftDoubleComplex* vzUpI,
							 cufftDoubleComplex* cZ,
							 double* vxGPU, 
							 double* vyGPU, 
							 double* vzGPU,
							 double* cGPU,
							 double* cPredictionGPU){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   

  //vxGPU[j] = 0.5 * ( (vxUpI[j].x / double(ncellsGPU)) + fetch_double(texVxGPU,j));
  //vyGPU[j] = 0.5 * ( (vxUpI[j].y / double(ncellsGPU)) + fetch_double(texVyGPU,j));
  //vzGPU[j] = 0.5 * ( (vzUpI[j].x / double(ncellsGPU)) + fetch_double(texVzGPU,j));
  //cPredictionGPU[j] = 0.5 * ( (vzUpI[j].y / double(ncellsGPU)) + cGPU[j]);


  vxGPU[j] = vxUpI[j].x / double(ncellsGPU) ;
  vyGPU[j] = vxUpI[j].y / double(ncellsGPU) ;
  vzGPU[j] = vzUpI[j].x / double(ncellsGPU) ;
  cPredictionGPU[j] = vzUpI[j].y / double(ncellsGPU) ;

  //vxGPU[j] = vxUpI[j].x / double(ncellsGPU) ;
  //vyGPU[j] = vyUpI[j].x / double(ncellsGPU) ;
  //vzGPU[j] = vzUpI[j].x / double(ncellsGPU) ;
  //cPredictionGPU[j] = cZ[j].x / double(ncellsGPU) ;
  

  
}













__global__ void calculateVelocityPredictionBinaryMixtureMidPoint(cufftDoubleComplex* vxUpI, 
								 cufftDoubleComplex* vyUpI,
								 cufftDoubleComplex* vzUpI,
								 cufftDoubleComplex* cZ,
								 double* vxGPU, 
								 double* vyGPU, 
								 double* vzGPU,
								 double* cGPU,
								 double* cPredictionGPU){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   

  vxGPU[j] = 0.5 * ( (vxUpI[j].x / double(ncellsGPU)) + fetch_double(texVxGPU,j));
  vyGPU[j] = 0.5 * ( (vxUpI[j].y / double(ncellsGPU)) + fetch_double(texVyGPU,j));
  vzGPU[j] = 0.5 * ( (vzUpI[j].x / double(ncellsGPU)) + fetch_double(texVzGPU,j));
  cPredictionGPU[j] = 0.5 * ( (vzUpI[j].y / double(ncellsGPU)) + cGPU[j]);

}





















__global__ void normalizeField(double* vxGPU, double* vyGPU, double* vzGPU){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  vxGPU[i] = vxGPU[i] / double(ncellsGPU);
  vyGPU[i] = vyGPU[i] / double(ncellsGPU);
  vzGPU[i] = vzGPU[i] / double(ncellsGPU);

}

__global__ void normalizeFieldComplex(cufftDoubleComplex* vxGPU, 
				      cufftDoubleComplex* vyGPU, 
				      cufftDoubleComplex* vzGPU){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  vxGPU[i].x = vxGPU[i].x / double(ncellsGPU);
  vyGPU[i].x = vyGPU[i].x / double(ncellsGPU);
  vzGPU[i].x = vzGPU[i].x / double(ncellsGPU);

  vxGPU[i].y = vxGPU[i].y / double(ncellsGPU);
  vyGPU[i].y = vyGPU[i].y / double(ncellsGPU);
  vzGPU[i].y = vzGPU[i].y / double(ncellsGPU);

}

__global__ void normalizeFieldComplexRealPart(cufftDoubleComplex* vxGPU, 
					      cufftDoubleComplex* vyGPU, 
					      cufftDoubleComplex* vzGPU){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  vxGPU[i].x = vxGPU[i].x / double(ncellsGPU);
  vyGPU[i].x = vyGPU[i].x / double(ncellsGPU);
  vzGPU[i].x = vzGPU[i].x / double(ncellsGPU);

}


__global__ void kernelShift(cufftDoubleComplex *WxZ, 
			    cufftDoubleComplex *WyZ,
			    cufftDoubleComplex *WzZ,
			    prefactorsFourier *pF,
			    int sign){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double pi = 4. * atan(1.);

  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  cufftDoubleComplex aux;
    
  //aux.x = WxZ[i].x * cos(pi*kx/double(mxGPU)) - sign * WxZ[i].y * sin(pi*kx/double(mxGPU));
  //aux.y = WxZ[i].y * cos(pi*kx/double(mxGPU)) + sign * WxZ[i].x * sin(pi*kx/double(mxGPU));
  aux.x = WxZ[i].x * pF->expKx[kx].x - sign * WxZ[i].y * pF->expKx[kx].y;
  aux.y = WxZ[i].y * pF->expKx[kx].x + sign * WxZ[i].x * pF->expKx[kx].y;
  WxZ[i].x = aux.x;
  WxZ[i].y = aux.y;

  //aux.x = WyZ[i].x * cos(pi*ky/double(myGPU)) - sign * WyZ[i].y * sin(pi*ky/double(myGPU));
  //aux.y = WyZ[i].y * cos(pi*ky/double(myGPU)) + sign * WyZ[i].x * sin(pi*ky/double(myGPU));
  aux.x = WyZ[i].x * pF->expKy[ky].x - sign * WyZ[i].y * pF->expKy[ky].y;
  aux.y = WyZ[i].y * pF->expKy[ky].x + sign * WyZ[i].x * pF->expKy[ky].y;
  WyZ[i].x = aux.x;
  WyZ[i].y = aux.y;

  //aux.x = WzZ[i].x * cos(pi*kz/double(mzGPU)) - sign * WzZ[i].y * sin(pi*kz/double(mzGPU));
  //aux.y = WzZ[i].y * cos(pi*kz/double(mzGPU)) + sign * WzZ[i].x * sin(pi*kz/double(mzGPU));
  aux.x = WzZ[i].x * pF->expKz[kz].x - sign * WzZ[i].y * pF->expKz[kz].y;
  aux.y = WzZ[i].y * pF->expKz[kz].x + sign * WzZ[i].x * pF->expKz[kz].y;
  WzZ[i].x = aux.x;
  WzZ[i].y = aux.y;


}


























__global__ void kernelShiftBinaryMixture(cufftDoubleComplex *WxZ, 
					 cufftDoubleComplex *WyZ,
					 cufftDoubleComplex *WzZ,
					 cufftDoubleComplex *cZ,
					 prefactorsFourier *pF,
					 int sign){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  //double pi = 4. * atan(1.);

  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  if(kz>(mzGPU/2)) return;

  int kxC, kyC, kzC, j;
  kzC = (mzGPU-kz) % mzGPU;
  kxC = (mxGPU-kx) % mxGPU;
  kyC = (myGPU-ky) % myGPU;

  if( (kz==kzC) && (ky>myGPU/2) ) return;
  if( (kz==kzC) && (ky==kyC) && (kx>mxGPU/2) ) return;  
  
  cufftDoubleComplex w, wC;
  double rep, rem, ip, im;

  //
  //
  //Cell (kx,ky,kz) and complementary cell (kxC, kyC, kzC)
  j = kxC + kyC * mxGPU + kzC * mxGPU*myGPU;

  //Extract x and y componenet
  rep = 0.5 * (WxZ[i].x + WxZ[j].x);
  rem = 0.5 * (WxZ[i].x - WxZ[j].x);
  ip  = 0.5 * (WxZ[i].y + WxZ[j].y);
  im  = 0.5 * (WxZ[i].y - WxZ[j].y);

  //Write x componenet
  w.x = rep;
  w.y = im;
  wC.x = rep;
  wC.y = -im;
  
  WxZ[i].x = w.x  * pF->expKx[kx].x  - sign * w.y  * pF->expKx[kx].y;
  WxZ[i].y = w.y  * pF->expKx[kx].x  + sign * w.x  * pF->expKx[kx].y;
  WxZ[j].x = wC.x * pF->expKx[kxC].x - sign * wC.y * pF->expKx[kxC].y;
  WxZ[j].y = wC.y * pF->expKx[kxC].x + sign * wC.x * pF->expKx[kxC].y;
  
  //Write y component
  w.x = ip;
  w.y = -rem;
  wC.x = ip;
  wC.y = rem;

  WyZ[i].x = w.x  * pF->expKy[ky].x  - sign * w.y  * pF->expKy[ky].y;
  WyZ[i].y = w.y  * pF->expKy[ky].x  + sign * w.x  * pF->expKy[ky].y;
  WyZ[j].x = wC.x * pF->expKy[kyC].x - sign * wC.y * pF->expKy[kyC].y;
  WyZ[j].y = wC.y * pF->expKy[kyC].x + sign * wC.x * pF->expKy[kyC].y;



  //Extract z and concentration componenet
  rep = 0.5 * (WzZ[i].x + WzZ[j].x);
  rem = 0.5 * (WzZ[i].x - WzZ[j].x);
  ip  = 0.5 * (WzZ[i].y + WzZ[j].y);
  im  = 0.5 * (WzZ[i].y - WzZ[j].y);

  //Write z component
  w.x = rep;
  w.y = im;
  wC.x = rep;
  wC.y = -im;

  WzZ[i].x = w.x  * pF->expKz[kz].x  - sign * w.y  * pF->expKz[kz].y;
  WzZ[i].y = w.y  * pF->expKz[kz].x  + sign * w.x  * pF->expKz[kz].y;
  WzZ[j].x = wC.x * pF->expKz[kzC].x - sign * wC.y * pF->expKz[kzC].y;
  WzZ[j].y = wC.y * pF->expKz[kzC].x + sign * wC.x * pF->expKz[kzC].y;
  
  //Write c component
  w.x = ip;
  w.y = -rem;
  wC.x = ip;
  wC.y = rem;  

  cZ[i].x = w.x    ;
  cZ[i].y = w.y    ;
  cZ[j].x = wC.x   ;
  cZ[j].y = wC.y   ;



}


















__global__ void kernelShiftBinaryMixture_2(cufftDoubleComplex *WxZ, 
					   cufftDoubleComplex *WyZ,
					   cufftDoubleComplex *WzZ,
					   cufftDoubleComplex *cZ,
					   prefactorsFourier *pF,
					   int sign){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;


  cufftDoubleComplex w, wC;
  //Write x and y componenet
  w.x = WxZ[i].x * pF->expKx[kx].x - sign * WxZ[i].y * pF->expKx[kx].y;
  w.y = WxZ[i].y * pF->expKx[kx].x + sign * WxZ[i].x * pF->expKx[kx].y;

  wC.x = WyZ[i].x * pF->expKy[ky].x - sign * WyZ[i].y * pF->expKy[ky].y;
  wC.y = WyZ[i].y * pF->expKy[ky].x + sign * WyZ[i].x * pF->expKy[ky].y;

  WxZ[i].x = w.x - wC.y;
  WxZ[i].y = w.y + wC.x;



  //Write z and concentration component
  w.x  = WzZ[i].x * pF->expKz[kz].x  - sign * WzZ[i].y * pF->expKz[kz].y;
  w.y  = WzZ[i].y * pF->expKz[kz].x  + sign * WzZ[i].x * pF->expKz[kz].y;

  //wC.x = WyZ[i].x * pF->expKy[ky].x - sign * WyZ[i].y * pF->expKy[ky].y;
  //wC.y = WyZ[i].y * pF->expKy[ky].x + sign * WyZ[i].x * pF->expKy[ky].y;

  WzZ[i].x = w.x - cZ[i].y;
  WzZ[i].y = w.y + cZ[i].x;


}






















__global__ void kernelShiftBinaryMixture_3(cufftDoubleComplex *WxZ, 
					   cufftDoubleComplex *WyZ,
					   cufftDoubleComplex *WzZ,
					   cufftDoubleComplex *cZ,
					   prefactorsFourier *pF,
					   int sign){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double pi = 4. * atan(1.);

  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  cufftDoubleComplex aux;
    
  aux.x = WxZ[i].x * pF->expKx[kx].x - sign * WxZ[i].y * pF->expKx[kx].y;
  aux.y = WxZ[i].y * pF->expKx[kx].x + sign * WxZ[i].x * pF->expKx[kx].y;
  WxZ[i].x = aux.x;
  WxZ[i].y = aux.y;

  aux.x = WyZ[i].x * pF->expKy[ky].x - sign * WyZ[i].y * pF->expKy[ky].y;
  aux.y = WyZ[i].y * pF->expKy[ky].x + sign * WyZ[i].x * pF->expKy[ky].y;
  WyZ[i].x = aux.x;
  WyZ[i].y = aux.y;

  aux.x = WzZ[i].x * pF->expKz[kz].x - sign * WzZ[i].y * pF->expKz[kz].y;
  aux.y = WzZ[i].y * pF->expKz[kz].x + sign * WzZ[i].x * pF->expKz[kz].y;
  WzZ[i].x = aux.x;
  WzZ[i].y = aux.y;


}



















__global__ void doubleComplexToDoubleNormalizedAndNormalizeComplex(cufftDoubleComplex* vxUpI, 
								   cufftDoubleComplex* vyUpI,
								   cufftDoubleComplex* vzUpI, 
								   double* vxGPU, 
								   double* vyGPU, 
								   double* vzGPU){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j>=ncellsGPU) return;   
  
  vxGPU[j] = vxUpI[j].x / double(ncellsGPU);
  vyGPU[j] = vyUpI[j].x / double(ncellsGPU);
  vzGPU[j] = vzUpI[j].x / double(ncellsGPU);

  vxUpI[j].x = vxUpI[j].x / double(ncellsGPU);
  vyUpI[j].x = vyUpI[j].x / double(ncellsGPU);
  vzUpI[j].x = vzUpI[j].x / double(ncellsGPU);

}




__global__ void setArrayToZeroInput(double* vxboundaryPredictionGPU,
				    double* vyboundaryPredictionGPU,
				    double* vzboundaryPredictionGPU,
				    const double input){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=npGPU) return;   

  vxboundaryPredictionGPU[i] = 0;//input;
  vyboundaryPredictionGPU[i] = 0;//input;
  vzboundaryPredictionGPU[i] = 0;//input;

}



__global__ void setFieldToZeroInput(double* vxGPU,
				    double* vyGPU,
				    double* vzGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  vxGPU[i] = 0;//input;
  vyGPU[i] = 0;//input;
  vzGPU[i] = 0;//input;

}


