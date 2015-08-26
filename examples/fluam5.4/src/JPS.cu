// Filename: JPS.cu
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


__global__ void spreadVector(double rx, //Point
			     double ry,
			     double rz,
			     double ux, //Vector to spread
			     double uy,
			     double uz,
			     double* rxcellGPU, 
			     double* rycellGPU, 
			     double* rzcellGPU,
			     double* vxGPU, 
			     double* vyGPU, 
			     double* vzGPU){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   


  double r;
  double dlx;
  double dly;
  double dlz;

  //Spreading component X
  r =  (rx - rxcellGPU[i] - dxGPU*0.5);
  r =  invdxGPU * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  dlx  = delta(r);

  r =  (ry - rycellGPU[i]);
  r =  invdyGPU * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  dly  = delta(r);

  r =  (rz - rzcellGPU[i]);
  r  = invdzGPU * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  dlz  = delta(r);

  vxGPU[i] = dlx  * dly  * dlz  * ux;

  //Spreading component Y
  r =  (rx - rxcellGPU[i]);
  r =  invdxGPU * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  dlx  = delta(r);

  r =  (ry - rycellGPU[i] - dyGPU*0.5);
  r =  invdyGPU * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  dly  = delta(r);

  r =  (rz - rzcellGPU[i]);
  r  = invdzGPU * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  dlz  = delta(r);

  vyGPU[i] = dlx * dly * dlz * uy;

  //Spreading component Z
  r =  (rx - rxcellGPU[i]);
  r =  invdxGPU * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  dlx  = delta(r);

  r =  (ry - rycellGPU[i]);
  r =  invdyGPU * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  dly  = delta(r);

  r =  (rz - rzcellGPU[i] - dzGPU*0.5);
  r  = invdzGPU * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  dlz  = delta(r);

  vzGPU[i] = dlx * dly * dlz * uz;






  return;
}














__global__ void kernelConstructWTestJPS(int axis,
					double *vxGPU,
					double *vyGPU,
					double *vzGPU,
					cufftDoubleComplex *vxZ,
					cufftDoubleComplex *vyZ,
					cufftDoubleComplex *vzZ){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  if(axis==0){ //axis x
    vxZ[i].x = vxGPU[i];
    vxZ[i].y = 0;
    vyZ[i].x = 0;
    vyZ[i].y = 0;
    vzZ[i].x = 0;
    vzZ[i].y = 0;
  }
  else if(axis==1){ //axis y
    vxZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].x = vyGPU[i];
    vyZ[i].y = 0;
    vzZ[i].x = 0;
    vzZ[i].y = 0;
  }
  else if(axis==2){ //axis z
    vxZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].x = 0;
    vyZ[i].y = 0;
    vzZ[i].x = vzGPU[i];
    vzZ[i].y = 0;
  }


  return;
}











//Interpolate fluid velocity to particle position
__global__ void interpolateField2(double* rxcellGPU,
				  double* rycellGPU,
				  double* rzcellGPU,
				  cufftDoubleComplex* vxZ,
				  cufftDoubleComplex* vyZ,
				  cufftDoubleComplex* vzZ,
				  double rx, //Point
				  double ry,
				  double rz,
				  double* vxboundaryGPU, //Final result JPS
				  double* vyboundaryGPU,
				  double* vzboundaryGPU){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(1)) return;   

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double r, rp, rm;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;

  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * myGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icelx);
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecino5 = tex1Dfetch(texvecino5GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelx);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelx);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelx);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelx);
  int vecinopxpxpypz = tex1Dfetch(texvecino3GPU, vecinopxpypz);
  int vecinopxpxpymz = tex1Dfetch(texvecino3GPU, vecinopxpymz);
  int vecinopxpxmypz = tex1Dfetch(texvecino3GPU, vecinopxmypz);
  int vecinopxpxmymz = tex1Dfetch(texvecino3GPU, vecinopxmymz);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  int vecinopxpxpz   = tex1Dfetch(texvecino3GPU, vecinopxpz);
  int vecinopxpxmz   = tex1Dfetch(texvecino3GPU, vecinopxmz);
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  invdxGPU * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = invdxGPU * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = invdxGPU * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(r);
  dlxp = delta(rp);
  dlxm = delta(rm);

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  invdyGPU * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = invdyGPU * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = invdyGPU * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(r);
  dlyp = delta(rp);
  dlym = delta(rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = invdzGPU * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = invdzGPU * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = invdzGPU * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(r);
  dlzp = delta(rp);
  dlzm = delta(rm);
  
  double v = dlxm * dlym * dlzm * vxZ[vecinomxmymz].x +
    dlxm * dlym * dlz  * vxZ[vecinomxmy].x +
    dlxm * dlym * dlzp * vxZ[vecinomxmypz].x +
    dlxm * dly  * dlzm * vxZ[vecinomxmz].x + 
    dlxm * dly  * dlz  * vxZ[vecino2].x +
    dlxm * dly  * dlzp * vxZ[vecinomxpz].x +
    dlxm * dlyp * dlzm * vxZ[vecinomxpymz].x +
    dlxm * dlyp * dlz  * vxZ[vecinomxpy].x +
    dlxm * dlyp * dlzp * vxZ[vecinomxpypz].x +
    dlx  * dlym * dlzm * vxZ[vecinomymz].x +
    dlx  * dlym * dlz  * vxZ[vecino1].x +
    dlx  * dlym * dlzp * vxZ[vecinomypz].x + 
    dlx  * dly  * dlzm * vxZ[vecino0].x + 
    dlx  * dly  * dlz  * vxZ[icelx].x + 
    dlx  * dly  * dlzp * vxZ[vecino5].x + 
    dlx  * dlyp * dlzm * vxZ[vecinopymz].x + 
    dlx  * dlyp * dlz  * vxZ[vecino4].x + 
    dlx  * dlyp * dlzp * vxZ[vecinopypz].x + 
    dlxp * dlym * dlzm * vxZ[vecinopxmymz].x + 
    dlxp * dlym * dlz  * vxZ[vecinopxmy].x + 
    dlxp * dlym * dlzp * vxZ[vecinopxmypz].x +
    dlxp * dly  * dlzm * vxZ[vecinopxmz].x + 
    dlxp * dly  * dlz  * vxZ[vecino3].x +
    dlxp * dly  * dlzp * vxZ[vecinopxpz].x +
    dlxp * dlyp * dlzm * vxZ[vecinopxpymz].x +
    dlxp * dlyp * dlz  * vxZ[vecinopxpy].x +
    dlxp * dlyp * dlzp * vxZ[vecinopxpypz].x;
  

  vxboundaryGPU[nboundaryGPU+i] = v / ncellsGPU * volumeParticleGPU ; 
      
  //VELOCITY IN THE Y DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icely);
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecino5 = tex1Dfetch(texvecino5GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icely);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icely);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icely);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icely);  
  //DEFINE MORE NEIGHBORS
  int vecinopymxpymz = tex1Dfetch(texvecino4GPU, vecinomxpymz);
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopymxpypz = tex1Dfetch(texvecino4GPU, vecinomxpypz);
  int vecinopypymz   = tex1Dfetch(texvecino4GPU, vecinopymz);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypypz   = tex1Dfetch(texvecino4GPU, vecinopypz);
  int vecinopypxpymz = tex1Dfetch(texvecino4GPU, vecinopxpymz);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);
  int vecinopypxpypz = tex1Dfetch(texvecino4GPU, vecinopxpypz);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  invdxGPU * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = invdxGPU * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = invdxGPU * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(r);
  dlxp = delta(rp);
  dlxm = delta(rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  invdyGPU * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = invdyGPU * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = invdyGPU * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(r);
  dlyp = delta(rp);
  dlym = delta(rm);

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = invdzGPU * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = invdzGPU * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = invdzGPU * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(r);
  dlzp = delta(rp);
  dlzm = delta(rm);

  
  v = dlxm * dlym * dlzm * vyZ[vecinomxmymz].x +
    dlxm * dlym * dlz  * vyZ[vecinomxmy].x +
    dlxm * dlym * dlzp * vyZ[vecinomxmypz].x +
    dlxm * dly  * dlzm * vyZ[vecinomxmz].x + 
    dlxm * dly  * dlz  * vyZ[vecino2].x +
    dlxm * dly  * dlzp * vyZ[vecinomxpz].x +
    dlxm * dlyp * dlzm * vyZ[vecinomxpymz].x +
    dlxm * dlyp * dlz  * vyZ[vecinomxpy].x +
    dlxm * dlyp * dlzp * vyZ[vecinomxpypz].x +
    dlx  * dlym * dlzm * vyZ[vecinomymz].x +
    dlx  * dlym * dlz  * vyZ[vecino1].x +
    dlx  * dlym * dlzp * vyZ[vecinomypz].x + 
    dlx  * dly  * dlzm * vyZ[vecino0].x + 
    dlx  * dly  * dlz  * vyZ[icely].x + 
    dlx  * dly  * dlzp * vyZ[vecino5].x + 
    dlx  * dlyp * dlzm * vyZ[vecinopymz].x + 
    dlx  * dlyp * dlz  * vyZ[vecino4].x + 
    dlx  * dlyp * dlzp * vyZ[vecinopypz].x + 
    dlxp * dlym * dlzm * vyZ[vecinopxmymz].x + 
    dlxp * dlym * dlz  * vyZ[vecinopxmy].x + 
    dlxp * dlym * dlzp * vyZ[vecinopxmypz].x +
    dlxp * dly  * dlzm * vyZ[vecinopxmz].x + 
    dlxp * dly  * dlz  * vyZ[vecino3].x +
    dlxp * dly  * dlzp * vyZ[vecinopxpz].x +
    dlxp * dlyp * dlzm * vyZ[vecinopxpymz].x +
    dlxp * dlyp * dlz  * vyZ[vecinopxpy].x +
    dlxp * dlyp * dlzp * vyZ[vecinopxpypz].x;
									  
  vyboundaryGPU[nboundaryGPU+i] = v / ncellsGPU * volumeParticleGPU ;

  //VELOCITY IN THE Z DIRECTION
  vecino0 = tex1Dfetch(texvecino0GPU, icelz);
  vecino1 = tex1Dfetch(texvecino1GPU, icelz);
  vecino2 = tex1Dfetch(texvecino2GPU, icelz);
  vecino3 = tex1Dfetch(texvecino3GPU, icelz);
  vecino4 = tex1Dfetch(texvecino4GPU, icelz);
  vecino5 = tex1Dfetch(texvecino5GPU, icelz);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelz);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelz);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelz);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelz);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelz);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelz);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelz);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelz);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelz);  
  //DEFINE MORE NEIGHBORS
  int vecinopzmxmypz = tex1Dfetch(texvecino5GPU, vecinomxmypz);
  int vecinopzmxpz   = tex1Dfetch(texvecino5GPU, vecinomxpz);
  int vecinopzmxpypz = tex1Dfetch(texvecino5GPU, vecinomxpypz);
  int vecinopzmypz   = tex1Dfetch(texvecino5GPU, vecinomypz);
  int vecinopzpz     = tex1Dfetch(texvecino5GPU, vecino5);
  int vecinopzpypz   = tex1Dfetch(texvecino5GPU, vecinopypz);
  int vecinopzpxmypz = tex1Dfetch(texvecino5GPU, vecinopxmypz);
  int vecinopzpxpz   = tex1Dfetch(texvecino5GPU, vecinopxpz);
  int vecinopzpxpypz = tex1Dfetch(texvecino5GPU, vecinopxpypz);

  r =  (rx - rxcellGPU[icelz]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  invdxGPU * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = invdxGPU * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = invdxGPU * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(r);
  dlxp = delta(rp);
  dlxm = delta(rm);

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  invdyGPU * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = invdyGPU * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = invdyGPU * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(r);
  dlyp = delta(rp);
  dlym = delta(rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = invdzGPU * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = invdzGPU * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = invdzGPU * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = delta(r);
  dlzp = delta(rp);
  dlzm = delta(rm);


  v = dlxm * dlym * dlzm * vzZ[vecinomxmymz].x +
    dlxm * dlym * dlz  * vzZ[vecinomxmy].x +
    dlxm * dlym * dlzp * vzZ[vecinomxmypz].x +
    dlxm * dly  * dlzm * vzZ[vecinomxmz].x + 
    dlxm * dly  * dlz  * vzZ[vecino2].x +
    dlxm * dly  * dlzp * vzZ[vecinomxpz].x +
    dlxm * dlyp * dlzm * vzZ[vecinomxpymz].x +
    dlxm * dlyp * dlz  * vzZ[vecinomxpy].x +
    dlxm * dlyp * dlzp * vzZ[vecinomxpypz].x +
    dlx  * dlym * dlzm * vzZ[vecinomymz].x +
    dlx  * dlym * dlz  * vzZ[vecino1].x +
    dlx  * dlym * dlzp * vzZ[vecinomypz].x + 
    dlx  * dly  * dlzm * vzZ[vecino0].x + 
    dlx  * dly  * dlz  * vzZ[icelz].x + 
    dlx  * dly  * dlzp * vzZ[vecino5].x + 
    dlx  * dlyp * dlzm * vzZ[vecinopymz].x + 
    dlx  * dlyp * dlz  * vzZ[vecino4].x + 
    dlx  * dlyp * dlzp * vzZ[vecinopypz].x + 
    dlxp * dlym * dlzm * vzZ[vecinopxmymz].x + 
    dlxp * dlym * dlz  * vzZ[vecinopxmy].x + 
    dlxp * dlym * dlzp * vzZ[vecinopxmypz].x +
    dlxp * dly  * dlzm * vzZ[vecinopxmz].x + 
    dlxp * dly  * dlz  * vzZ[vecino3].x +
    dlxp * dly  * dlzp * vzZ[vecinopxpz].x +
    dlxp * dlyp * dlzm * vzZ[vecinopxpymz].x +
    dlxp * dlyp * dlz  * vzZ[vecinopxpy].x +
    dlxp * dlyp * dlzp * vzZ[vecinopxpypz].x;
  
  
  vzboundaryGPU[nboundaryGPU+i] = v / ncellsGPU * volumeParticleGPU ;




}


