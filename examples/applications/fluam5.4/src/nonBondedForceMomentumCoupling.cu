// Filename: nonBondedForceCompressibleParticles.cu
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


__global__ void nonBondedForceMomentumCoupling(double* rxcellGPU, 
					       double* rycellGPU, 
					       double* rzcellGPU,
					       double* vxboundaryGPU, 
					       double* vyboundaryGPU, 
					       double* vzboundaryGPU,
					       double* densityGPU,
					       double* fxboundaryGPU, 
					       double* fyboundaryGPU, 
					       double* fzboundaryGPU,
					       particlesincell* pc, 
					       int* errorKernel,
					       double* saveForceX, 
					       double* saveForceY, 
					       double* saveForceZ){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double fx = 0.;
  double fy = 0.;
  double fz = 0.;
  double f;
  
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  //double rx = rxboundaryPredictionGPU[nboundaryGPU+i];
  //double ry = ryboundaryPredictionGPU[nboundaryGPU+i];
  //double rz = rzboundaryPredictionGPU[nboundaryGPU+i];
  ///////////////////
  //fx = -0.0025 * rx;
  //fy = -0.0025 * ry;
  //fz = -0.0025 * rz;

  //double rxij, ryij, rzij, r2;

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;
  
  //int icel;
  double r, rp, rm;

  /*{
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }*/
  

  
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

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

  int np = atomicAdd(&pc->countparticlesincellX[icelx],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellstGPU*np+icelx] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellY[icely],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[2]=np;
    return;
  }
  pc->partincellY[ncellstGPU*np+icely] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellZ[icelz],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[3]=np;
    return;
  }
  pc->partincellZ[ncellstGPU*np+icelz] = nboundaryGPU+i;

  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;

  //FORCE IN THE X DIRECTION
  //vecino0 = in->vecino0GPU[icelx];
  vecino0 = tex1Dfetch(texvecino0GPU, icelx);
  //vecino1 = in->vecino1GPU[icelx];
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  //vecino2 = in->vecino2GPU[icelx];
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  //vecino3 = in->vecino3GPU[icelx];
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  //vecino4 = in->vecino4GPU[icelx];
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  //vecino5 = in->vecino5GPU[icelx];
  vecino5 = tex1Dfetch(texvecino5GPU, icelx);
  //vecinopxpy = in->vecinopxpyGPU[icelx];
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  //vecinopxmy = in->vecinopxmyGPU[icelx];  
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  //vecinopxpz = in->vecinopxpzGPU[icelx];
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelx);
  //vecinopxmz = in->vecinopxmzGPU[icelx];
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelx);
  //vecinomxpy = in->vecinomxpyGPU[icelx];
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  //vecinomxmy = in->vecinomxmyGPU[icelx];
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  //vecinomxpz = in->vecinomxpzGPU[icelx];
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  //vecinomxmz = in->vecinomxmzGPU[icelx];
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelx);
  //vecinopypz = in->vecinopypzGPU[icelx];
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  //vecinopymz = in->vecinopymzGPU[icelx];
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  //vecinomypz = in->vecinomypzGPU[icelx];
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  //vecinomymz = in->vecinomymzGPU[icelx];
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelx);
  //vecinopxpypz = in->vecinopxpypzGPU[icelx];
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  //vecinopxpymz = in->vecinopxpymzGPU[icelx];
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  //vecinopxmypz = in->vecinopxmypzGPU[icelx];
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  //vecinopxmymz = in->vecinopxmymzGPU[icelx];
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  //vecinomxpypz = in->vecinomxpypzGPU[icelx];
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  //vecinomxpymz = in->vecinomxpymzGPU[icelx];
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  //vecinomxmypz = in->vecinomxmypzGPU[icelx];
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  //vecinomxmymz = in->vecinomxmymzGPU[icelx];
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
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);


  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  double v = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomymz]) * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecino1]) * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinomypz]) * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecino0]) * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[icelx]) * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecino5]) * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinopymz]) * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecino4]) * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopypz]) * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecinopxmymz]) * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[vecinopxmy]) * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecinopxmypz]) * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[vecinopxmz]) * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icelx]        + densityGPU[vecino3]) * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopxpz]) * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecinopxpymz]) * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopxpy]) * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopxpypz]) * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxpxmymz]) * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecinopxpxmy]) * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopxpxmypz]) * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecinopxpxmz]) * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpx]) * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz] + densityGPU[vecinopxpxpz]) * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopxpxpymz]) * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy] + densityGPU[vecinopxpxpy]) * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopxpxpypz]) * fetch_double(texVxGPU,vecinopxpypz);


  double rho = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomymz]) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecino1]) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinomypz]) +
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecino0]) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[icelx]) +
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecino5]) +
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinopymz]) +
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecino4]) +
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopypz]) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecinopxmymz]) +
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[vecinopxmy]) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecinopxmypz]) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[vecinopxmz]) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icelx]        + densityGPU[vecino3]) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopxpz]) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecinopxpymz]) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopxpy]) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopxpypz]) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxpxmymz]) +
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecinopxpxmy]) + 
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopxpxmypz]) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecinopxpxmz]) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpx]) + 
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz] + densityGPU[vecinopxpxpz]) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopxpxpymz]) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy] + densityGPU[vecinopxpxpy]) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopxpxpypz]);
  
  double u, uold, fact;

  uold = vxboundaryGPU[nboundaryGPU+i];
  //fact = volumeParticleGPU * volumeGPU * rho;
  //u = uold + (fact * (v-uold) + fx*dtGPU)/(massParticleGPU + fact);
  //vxboundaryGPU[nboundaryGPU+i] = u;
  //saveForceX[i] = massParticleGPU * fact * (uold - v) / ((massParticleGPU + fact) * dtGPU);
  //f = (v - u) * volumeParticleGPU;


  fact = volumeParticleGPU * volumeGPU;
  u = uold + (fact * (v-rho*uold) + fx*dtGPU) / (massParticleGPU + fact*rho);
  //u = uold + (fact * (v-densfluidGPU*uold) + fx*dtGPU) / (massParticleGPU + fact*densfluidGPU);
  vxboundaryGPU[nboundaryGPU+i] = u;
  f= (v - rho*u) * volumeParticleGPU;
  //f= (v - densfluidGPU*u) * volumeParticleGPU;


  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;
  
  
  //FORCE IN THE Y DIRECTION
  //vecino0 = in->vecino0GPU[icelx];
  vecino0 = tex1Dfetch(texvecino0GPU, icely);
  //vecino1 = in->vecino1GPU[icelx];
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  //vecino2 = in->vecino2GPU[icelx];
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  //vecino3 = in->vecino3GPU[icelx];
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  //vecino4 = in->vecino4GPU[icelx];
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  //vecino5 = in->vecino5GPU[icelx];
  vecino5 = tex1Dfetch(texvecino5GPU, icely);
  //vecinopxpy = in->vecinopxpyGPU[icelx];
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  //vecinopxmy = in->vecinopxmyGPU[icelx];  
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  //vecinopxpz = in->vecinopxpzGPU[icelx];
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icely);
  //vecinopxmz = in->vecinopxmzGPU[icelx];
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icely);
  //vecinomxpy = in->vecinomxpyGPU[icelx];
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  //vecinomxmy = in->vecinomxmyGPU[icelx];
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //vecinomxpz = in->vecinomxpzGPU[icelx];
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  //vecinomxmz = in->vecinomxmzGPU[icelx];
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icely);
  //vecinopypz = in->vecinopypzGPU[icelx];
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  //vecinopymz = in->vecinopymzGPU[icelx];
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  //vecinomypz = in->vecinomypzGPU[icelx];
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  //vecinomymz = in->vecinomymzGPU[icelx];
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icely);
  //vecinopxpypz = in->vecinopxpypzGPU[icelx];
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  //vecinopxpymz = in->vecinopxpymzGPU[icelx];
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  //vecinopxmypz = in->vecinopxmypzGPU[icelx];
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  //vecinopxmymz = in->vecinopxmymzGPU[icelx];
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  //vecinomxpypz = in->vecinomxpypzGPU[icelx];
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  //vecinomxpymz = in->vecinomxpymzGPU[icelx];
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  //vecinomxmypz = in->vecinomxmypzGPU[icelx];
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
  //vecinomxmymz = in->vecinomxmymzGPU[icelx];
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
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);


  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);


  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);



  v = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomxmz]) * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecino2]) * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinomxpz]) * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecinomxpymz]) * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[vecinomxpy]) * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecinomxpypz]) * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinopymxpymz]) * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecinopymxpy]) * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopymxpypz]) * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecino0]) * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[icely]) * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecino5]) * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[vecinopymz]) * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icely]        + densityGPU[vecino4]) * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopypz]) * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecinopypymz]) * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopypy]) * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopypypz]) * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxmz]) * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecino3]) * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopxpz]) * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecinopxpymz]) * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpy]) * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz]   + densityGPU[vecinopxpypz]) * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopypxpymz]) * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy]   + densityGPU[vecinopypxpy]) * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopypxpypz]) * fetch_double(texVyGPU,vecinopxpypz);

  rho = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomxmz]) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecino2]) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinomxpz]) +
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecinomxpymz]) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[vecinomxpy]) +
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecinomxpypz]) +
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinopymxpymz]) +
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecinopymxpy]) +
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopymxpypz]) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecino0]) +
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[icely]) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecino5]) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[vecinopymz]) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icely]        + densityGPU[vecino4]) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopypz]) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecinopypymz]) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopypy]) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopypypz]) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxmz]) +
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecino3]) +
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopxpz]) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecinopxpymz]) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpy]) + 
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz]   + densityGPU[vecinopxpypz]) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopypxpymz]) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy]   + densityGPU[vecinopypxpy]) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopypxpypz]);

  uold = vyboundaryGPU[nboundaryGPU+i];
  //fact = volumeParticleGPU * volumeGPU * rho;
  //u = uold + (fact * (v-uold) + fy * dtGPU)/(massParticleGPU + fact);
  //vyboundaryGPU[nboundaryGPU+i] = u;
  //saveForceY[i] = massParticleGPU * fact * (uold - v) / ((massParticleGPU + fact) * dtGPU);
  //f = (v - u) * volumeParticleGPU;

  //fact = volumeParticleGPU * volumeGPU;
  u = uold + (fact * (v-rho*uold) + fy*dtGPU) / (massParticleGPU + fact*rho);
  //u = uold + (fact * (v-densfluidGPU*uold) + fy*dtGPU) / (massParticleGPU + fact*densfluidGPU);
  vyboundaryGPU[nboundaryGPU+i] = u;
  f= (v - rho*u) * volumeParticleGPU;
  //f= (v - densfluidGPU*u) * volumeParticleGPU;


  offset = nboundaryGPU;
  fyboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;
  
  
  //FORCE IN THE Z DIRECTION
  //vecino0 = in->vecino0GPU[icelx];
  vecino0 = tex1Dfetch(texvecino0GPU, icelz);
  //vecino1 = in->vecino1GPU[icelx];
  vecino1 = tex1Dfetch(texvecino1GPU, icelz);
  //vecino2 = in->vecino2GPU[icelx];
  vecino2 = tex1Dfetch(texvecino2GPU, icelz);
  //vecino3 = in->vecino3GPU[icelx];
  vecino3 = tex1Dfetch(texvecino3GPU, icelz);
  //vecino4 = in->vecino4GPU[icelx];
  vecino4 = tex1Dfetch(texvecino4GPU, icelz);
  //vecino5 = in->vecino5GPU[icelx];
  vecino5 = tex1Dfetch(texvecino5GPU, icelz);
  //vecinopxpy = in->vecinopxpyGPU[icelx];
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelz);
  //vecinopxmy = in->vecinopxmyGPU[icelx];  
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelz);
  //vecinopxpz = in->vecinopxpzGPU[icelx];
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelz);
  //vecinopxmz = in->vecinopxmzGPU[icelx];
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelz);
  //vecinomxpy = in->vecinomxpyGPU[icelx];
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelz);
  //vecinomxmy = in->vecinomxmyGPU[icelx];
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelz);
  //vecinomxpz = in->vecinomxpzGPU[icelx];
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  //vecinomxmz = in->vecinomxmzGPU[icelx];
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelz);
  //vecinopypz = in->vecinopypzGPU[icelx];
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  //vecinopymz = in->vecinopymzGPU[icelx];
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  //vecinomypz = in->vecinomypzGPU[icelx];
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  //vecinomymz = in->vecinomymzGPU[icelx];
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelz);
  //vecinopxpypz = in->vecinopxpypzGPU[icelx];
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  //vecinopxpymz = in->vecinopxpymzGPU[icelx];
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  //vecinopxmypz = in->vecinopxmypzGPU[icelx];
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  //vecinopxmymz = in->vecinopxmymzGPU[icelx];
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  //vecinomxpypz = in->vecinomxpypzGPU[icelx];
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  //vecinomxpymz = in->vecinomxpymzGPU[icelx];
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  //vecinomxmypz = in->vecinomxmypzGPU[icelx];
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
  //vecinomxmymz = in->vecinomxmymzGPU[icelx];
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
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);


  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);


  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);



  v = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomxmy]) * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecinomxmypz]) * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinopzmxmypz]) * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecino2]) * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[vecinomxpz]) * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecinopzmxpz]) * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinomxpy]) * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecinomxpypz]) * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopzmxpypz]) * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecino1]) * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[vecinomypz]) * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecinopzmypz]) * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[icelz]) * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icelz]        + densityGPU[vecino5]) * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopzpz]) * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecino4]) * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopypz]) * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopzpypz]) * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxmy]) * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecinopxmypz]) * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopzpxmypz]) * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecino3]) * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpz]) * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz]   + densityGPU[vecinopzpxpz]) * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopxpy]) * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy]   + densityGPU[vecinopxpypz]) * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopzpxpypz]) * fetch_double(texVzGPU,vecinopxpypz);

  rho = dlxm * dlym * dlzm * 0.5 * (densityGPU[vecinomxmymz] + densityGPU[vecinomxmy]) +
    dlxm * dlym * dlz  * 0.5 * (densityGPU[vecinomxmy]   + densityGPU[vecinomxmypz]) +
    dlxm * dlym * dlzp * 0.5 * (densityGPU[vecinomxmypz] + densityGPU[vecinopzmxmypz]) + 
    dlxm * dly  * dlzm * 0.5 * (densityGPU[vecinomxmz]   + densityGPU[vecino2]) + 
    dlxm * dly  * dlz  * 0.5 * (densityGPU[vecino2]      + densityGPU[vecinomxpz]) + 
    dlxm * dly  * dlzp * 0.5 * (densityGPU[vecinomxpz]   + densityGPU[vecinopzmxpz]) + 
    dlxm * dlyp * dlzm * 0.5 * (densityGPU[vecinomxpymz] + densityGPU[vecinomxpy]) + 
    dlxm * dlyp * dlz  * 0.5 * (densityGPU[vecinomxpy]   + densityGPU[vecinomxpypz]) + 
    dlxm * dlyp * dlzp * 0.5 * (densityGPU[vecinomxpypz] + densityGPU[vecinopzmxpypz]) +
    dlx  * dlym * dlzm * 0.5 * (densityGPU[vecinomymz]   + densityGPU[vecino1]) + 
    dlx  * dlym * dlz  * 0.5 * (densityGPU[vecino1]      + densityGPU[vecinomypz]) +
    dlx  * dlym * dlzp * 0.5 * (densityGPU[vecinomypz]   + densityGPU[vecinopzmypz]) + 
    dlx  * dly  * dlzm * 0.5 * (densityGPU[vecino0]      + densityGPU[icelz]) + 
    dlx  * dly  * dlz  * 0.5 * (densityGPU[icelz]        + densityGPU[vecino5]) + 
    dlx  * dly  * dlzp * 0.5 * (densityGPU[vecino5]      + densityGPU[vecinopzpz]) + 
    dlx  * dlyp * dlzm * 0.5 * (densityGPU[vecinopymz]   + densityGPU[vecino4]) + 
    dlx  * dlyp * dlz  * 0.5 * (densityGPU[vecino4]      + densityGPU[vecinopypz]) + 
    dlx  * dlyp * dlzp * 0.5 * (densityGPU[vecinopypz]   + densityGPU[vecinopzpypz]) + 
    dlxp * dlym * dlzm * 0.5 * (densityGPU[vecinopxmymz] + densityGPU[vecinopxmy]) +
    dlxp * dlym * dlz  * 0.5 * (densityGPU[vecinopxmy]   + densityGPU[vecinopxmypz]) +
    dlxp * dlym * dlzp * 0.5 * (densityGPU[vecinopxmypz] + densityGPU[vecinopzpxmypz]) +
    dlxp * dly  * dlzm * 0.5 * (densityGPU[vecinopxmz]   + densityGPU[vecino3]) + 
    dlxp * dly  * dlz  * 0.5 * (densityGPU[vecino3]      + densityGPU[vecinopxpz]) + 
    dlxp * dly  * dlzp * 0.5 * (densityGPU[vecinopxpz]   + densityGPU[vecinopzpxpz]) +
    dlxp * dlyp * dlzm * 0.5 * (densityGPU[vecinopxpymz] + densityGPU[vecinopxpy]) +
    dlxp * dlyp * dlz  * 0.5 * (densityGPU[vecinopxpy]   + densityGPU[vecinopxpypz]) +
    dlxp * dlyp * dlzp * 0.5 * (densityGPU[vecinopxpypz] + densityGPU[vecinopzpxpypz]);

  uold = vzboundaryGPU[nboundaryGPU+i];
  //fact = volumeParticleGPU * volumeGPU * rho;
  //u = uold + (fact * (v-uold) + fz * dtGPU)/(massParticleGPU + fact);
  //vzboundaryGPU[nboundaryGPU+i] = u;
  //saveForceZ[i] = massParticleGPU * fact * (uold - v) / ((massParticleGPU + fact) * dtGPU);
  //f = (v - u) * volumeParticleGPU;

  //fact = volumeParticleGPU * volumeGPU;
  u = uold + (fact * (v-rho*uold) + fz*dtGPU) / (massParticleGPU + fact*rho);
  //u = uold + (fact * (v-densfluidGPU*uold) + fz*dtGPU) / (massParticleGPU + fact*densfluidGPU);
  vzboundaryGPU[nboundaryGPU+i] = u;
  f= (v - rho*u) * volumeParticleGPU;
  //f= (v - densfluidGPU*u) * volumeParticleGPU;

  offset = nboundaryGPU;
  fzboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = dlxp * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = dlxp * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = dlx  * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = dlx  * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzm * f;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzp * f;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlz  * f;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzm * f;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzp * f;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = dlxm * dly  * dlz  * f;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzm * f;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzp * f;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = dlxm * dlym * dlz  * f;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzm * f;

  
}
