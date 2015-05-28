

//FOR STEP 4
//Calculate \delta u^{n+1/2} and saved in vxboundaryPredictionGPU
__global__ void kernelCalculateDeltau(double* rxcellGPU,
				      double* rycellGPU,
				      double* rzcellGPU,
				      double* vxGPU,
				      double* vyGPU,
				      double* vzGPU,
				      double* rxboundaryGPU,
				      double* ryboundaryGPU,
				      double* rzboundaryGPU,
				      double* vxboundaryPredictionGPU,
				      double* vyboundaryPredictionGPU,
				      double* vzboundaryPredictionGPU){

 
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  //delta u^{n+1/2}
  double dux;
  double duy;
  double duz;

  //Particle position at time n+1/2
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

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


  //FIRST CALCULATE J^{n+1/2} * v^n
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





  //X DIRECTION
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


  double v = dlxm * dlym * dlzm * vxGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vxGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vxGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vxGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vxGPU[vecino2] +
    dlxm * dly  * dlzp * vxGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vxGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vxGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vxGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vxGPU[vecinomymz] +
    dlx  * dlym * dlz  * vxGPU[vecino1] +
    dlx  * dlym * dlzp * vxGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vxGPU[vecino0] + 
    dlx  * dly  * dlz  * vxGPU[icelx] + 
    dlx  * dly  * dlzp * vxGPU[vecino5] + 
    dlx  * dlyp * dlzm * vxGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vxGPU[vecino4] + 
    dlx  * dlyp * dlzp * vxGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vxGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vxGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vxGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vxGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vxGPU[vecino3] +
    dlxp * dly  * dlzp * vxGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vxGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vxGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vxGPU[vecinopxpypz];

  
  dux = v;


  //Y DIRECTION
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


  v = dlxm * dlym * dlzm * vyGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vyGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vyGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vyGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vyGPU[vecino2] +
    dlxm * dly  * dlzp * vyGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vyGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vyGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vyGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vyGPU[vecinomymz] +
    dlx  * dlym * dlz  * vyGPU[vecino1] +
    dlx  * dlym * dlzp * vyGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vyGPU[vecino0] + 
    dlx  * dly  * dlz  * vyGPU[icely] + 
    dlx  * dly  * dlzp * vyGPU[vecino5] + 
    dlx  * dlyp * dlzm * vyGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vyGPU[vecino4] + 
    dlx  * dlyp * dlzp * vyGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vyGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vyGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vyGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vyGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vyGPU[vecino3] +
    dlxp * dly  * dlzp * vyGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vyGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vyGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vyGPU[vecinopxpypz]; 

  
  duy = v;


  //Z DIRECTION
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

  v = dlxm * dlym * dlzm * vzGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vzGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vzGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vzGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vzGPU[vecino2] +
    dlxm * dly  * dlzp * vzGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vzGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vzGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vzGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vzGPU[vecinomymz] +
    dlx  * dlym * dlz  * vzGPU[vecino1] +
    dlx  * dlym * dlzp * vzGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vzGPU[vecino0] + 
    dlx  * dly  * dlz  * vzGPU[icelz] + 
    dlx  * dly  * dlzp * vzGPU[vecino5] + 
    dlx  * dlyp * dlzm * vzGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vzGPU[vecino4] + 
    dlx  * dlyp * dlzp * vzGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vzGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vzGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vzGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vzGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vzGPU[vecino3] +
    dlxp * dly  * dlzp * vzGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vzGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vzGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vzGPU[vecinopxpypz];

  duz = v;











  //SECOND CALCULATE J^{n} * v^n

  //Particle positon at time n
  rx = rxboundaryGPU[i];
  ry = ryboundaryGPU[i];
  rz = rzboundaryGPU[i];


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





  //X DIRECTION
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
  //DEFINE MORE NEIGHBORS
  vecinopxpxpypz = tex1Dfetch(texvecino3GPU, vecinopxpypz);
  vecinopxpxpymz = tex1Dfetch(texvecino3GPU, vecinopxpymz);
  vecinopxpxmypz = tex1Dfetch(texvecino3GPU, vecinopxmypz);
  vecinopxpxmymz = tex1Dfetch(texvecino3GPU, vecinopxmymz);
  vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  vecinopxpxpz   = tex1Dfetch(texvecino3GPU, vecinopxpz);
  vecinopxpxmz   = tex1Dfetch(texvecino3GPU, vecinopxmz);

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


  v = dlxm * dlym * dlzm * vxGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vxGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vxGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vxGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vxGPU[vecino2] +
    dlxm * dly  * dlzp * vxGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vxGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vxGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vxGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vxGPU[vecinomymz] +
    dlx  * dlym * dlz  * vxGPU[vecino1] +
    dlx  * dlym * dlzp * vxGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vxGPU[vecino0] + 
    dlx  * dly  * dlz  * vxGPU[icelx] + 
    dlx  * dly  * dlzp * vxGPU[vecino5] + 
    dlx  * dlyp * dlzm * vxGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vxGPU[vecino4] + 
    dlx  * dlyp * dlzp * vxGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vxGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vxGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vxGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vxGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vxGPU[vecino3] +
    dlxp * dly  * dlzp * vxGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vxGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vxGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vxGPU[vecinopxpypz];

  
  dux -= v;


  //Y DIRECTION
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
  vecinopymxpymz = tex1Dfetch(texvecino4GPU, vecinomxpymz);
  vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  vecinopymxpypz = tex1Dfetch(texvecino4GPU, vecinomxpypz);
  vecinopypymz   = tex1Dfetch(texvecino4GPU, vecinopymz);
  vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  vecinopypypz   = tex1Dfetch(texvecino4GPU, vecinopypz);
  vecinopypxpymz = tex1Dfetch(texvecino4GPU, vecinopxpymz);
  vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);
  vecinopypxpypz = tex1Dfetch(texvecino4GPU, vecinopxpypz);

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


  v = dlxm * dlym * dlzm * vyGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vyGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vyGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vyGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vyGPU[vecino2] +
    dlxm * dly  * dlzp * vyGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vyGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vyGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vyGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vyGPU[vecinomymz] +
    dlx  * dlym * dlz  * vyGPU[vecino1] +
    dlx  * dlym * dlzp * vyGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vyGPU[vecino0] + 
    dlx  * dly  * dlz  * vyGPU[icely] + 
    dlx  * dly  * dlzp * vyGPU[vecino5] + 
    dlx  * dlyp * dlzm * vyGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vyGPU[vecino4] + 
    dlx  * dlyp * dlzp * vyGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vyGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vyGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vyGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vyGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vyGPU[vecino3] +
    dlxp * dly  * dlzp * vyGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vyGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vyGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vyGPU[vecinopxpypz]; 

  
  duy -= v;


  //Z DIRECTION
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
  vecinopzmxmypz = tex1Dfetch(texvecino5GPU, vecinomxmypz);
  vecinopzmxpz   = tex1Dfetch(texvecino5GPU, vecinomxpz);
  vecinopzmxpypz = tex1Dfetch(texvecino5GPU, vecinomxpypz);
  vecinopzmypz   = tex1Dfetch(texvecino5GPU, vecinomypz);
  vecinopzpz     = tex1Dfetch(texvecino5GPU, vecino5);
  vecinopzpypz   = tex1Dfetch(texvecino5GPU, vecinopypz);
  vecinopzpxmypz = tex1Dfetch(texvecino5GPU, vecinopxmypz);
  vecinopzpxpz   = tex1Dfetch(texvecino5GPU, vecinopxpz);
  vecinopzpxpypz = tex1Dfetch(texvecino5GPU, vecinopxpypz);

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

  v = dlxm * dlym * dlzm * vzGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vzGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vzGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vzGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vzGPU[vecino2] +
    dlxm * dly  * dlzp * vzGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vzGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vzGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vzGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vzGPU[vecinomymz] +
    dlx  * dlym * dlz  * vzGPU[vecino1] +
    dlx  * dlym * dlzp * vzGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vzGPU[vecino0] + 
    dlx  * dly  * dlz  * vzGPU[icelz] + 
    dlx  * dly  * dlzp * vzGPU[vecino5] + 
    dlx  * dlyp * dlzm * vzGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vzGPU[vecino4] + 
    dlx  * dlyp * dlzp * vzGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vzGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vzGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vzGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vzGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vzGPU[vecino3] +
    dlxp * dly  * dlzp * vzGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vzGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vzGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vzGPU[vecinopxpypz];

  duz -= v;


  //THIRD, add (J^{n+1/2} - J^n) * v^n to
  //0.5*dt*nu * J^{n-1/2} * L * \Delta v^{n+1/2}
  vxboundaryPredictionGPU[i] += dux;
  vyboundaryPredictionGPU[i] += duy;
  vzboundaryPredictionGPU[i] += duz;


  //\delta u = J*L*Delta u
  //vxboundaryPredictionGPU[i] += 0;
  //vyboundaryPredictionGPU[i] += 0;
  //vzboundaryPredictionGPU[i] += 0;

  //\delta u = 0
  //vxboundaryPredictionGPU[i] = 0;
  //vyboundaryPredictionGPU[i] = 0;
  //vzboundaryPredictionGPU[i] = 0;




}
























//FOR STEP 5, CALCULATE \Delta p
//Calculate \Delta p and saved in vxboundaryGPU
__global__ void kernelCalculateDeltap(double* rxcellGPU,
				      double* rycellGPU,
				      double* rzcellGPU,
				      double* vxboundaryGPU,
				      double* vyboundaryGPU,
				      double* vzboundaryGPU,
				      double* vxboundaryPredictionGPU,
				      double* vyboundaryPredictionGPU,
				      double* vzboundaryPredictionGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   


  //Particle position at time n+1/2
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

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





  //X DIRECTION
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


  double v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);

  
  vxboundaryGPU[i] = massParticleGPU * ( vxboundaryGPU[i] - v - vxboundaryPredictionGPU[i] );


  //Y DIRECTION
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


  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz); 

  
  vyboundaryGPU[i] = massParticleGPU * ( vyboundaryGPU[i] - v - vyboundaryPredictionGPU[i] );


  //Z DIRECTION
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

  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);

  vzboundaryGPU[i] = massParticleGPU * ( vzboundaryGPU[i] - v - vzboundaryPredictionGPU[i] );


}
























//First, spread \Delta p, prefactor * S*{\Delta p}
__global__ void kernelSpreadDeltap(double* rxcellGPU,
				   double* rycellGPU,
				   double* rzcellGPU,
				   double* vxboundaryGPU,
				   double* vyboundaryGPU,
				   double* vzboundaryGPU,
				   double* fxboundaryGPU,
				   double* fyboundaryGPU,
				   double* fzboundaryGPU){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopypz, vecinopymz, vecinomypz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz;

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  double r, rp, rm;

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


  //SPREAD IN THE X DIRECTION
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
  //vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  //vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelx);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  //vecinomymz = tex1Dfetch(texvecinomymzGPU, icelx);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  //vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelx);
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

  //double fact = 1.5 * volumeParticleGPU *massParticleGPU / 
  //(1.5 * volumeParticleGPU * volumeGPU * densfluidGPU + massParticleGPU);

  double fact = 1.5 * volumeParticleGPU * densfluidGPU /
    (densfluidGPU * 
     (1.5*volumeParticleGPU*volumeGPU*densfluidGPU + massParticleGPU));
		  
  
  double SDeltap = fact * vxboundaryGPU[i] ;


  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlxp * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlxp * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx  * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx  * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlxm * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = dlxm * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzm * SDeltap;





  //SPREAD IN THE Y DIRECTION
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
  //vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  //vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icely);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  //vecinomymz = tex1Dfetch(texvecinomymzGPU, icely);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
  //vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icely);  
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


  SDeltap = fact * vyboundaryGPU[i] ;


  offset = nboundaryGPU;
  fyboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlxp * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlxp * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx  * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx  * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlxm * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = dlxm * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzm * SDeltap;





  //SPREAD IN THE Z DIRECTION
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
  //vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelz);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  //vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelz);
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  //vecinomymz = tex1Dfetch(texvecinomymzGPU, icelz);
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
  //vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelz);  
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


  SDeltap = fact * vzboundaryGPU[i] ;

  offset = nboundaryGPU;
  fzboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = dlxp * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = dlxp * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = dlx  * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = dlx  * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = dlxm * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = dlxm * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzm * SDeltap;





}












//Store \Delta \tilde{ v } in vxZ.y
__global__ void saveDeltaTildev(cufftDoubleComplex* vxZ,
				cufftDoubleComplex* vyZ,
				cufftDoubleComplex* vzZ){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  vxZ[i].y = vxZ[i].x / ncellsGPU;
  vyZ[i].y = vyZ[i].x / ncellsGPU;
  vzZ[i].y = vzZ[i].x / ncellsGPU;


}



















//First, spread S*(\Delta p - m_e*J*\Delta \tilde{ v })
__global__ void kernelSpreadDeltapMinusJTildev(double* rxcellGPU,
					       double* rycellGPU,
					       double* rzcellGPU,
					       cufftDoubleComplex* vxZ,
					       cufftDoubleComplex* vyZ,
					       cufftDoubleComplex* vzZ,
					       double* vxboundaryGPU,
					       double* vyboundaryGPU,
					       double* vzboundaryGPU,
					       double* fxboundaryGPU,
					       double* fyboundaryGPU,
					       double* fzboundaryGPU){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  double r, rp, rm;

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


  //SPREAD IN THE X DIRECTION
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

  
  double tildev = dlxm * dlym * dlzm * vxZ[vecinomxmymz].y +
    dlxm * dlym * dlz  * vxZ[vecinomxmy].y +
    dlxm * dlym * dlzp * vxZ[vecinomxmypz].y +
    dlxm * dly  * dlzm * vxZ[vecinomxmz].y + 
    dlxm * dly  * dlz  * vxZ[vecino2].y +
    dlxm * dly  * dlzp * vxZ[vecinomxpz].y +
    dlxm * dlyp * dlzm * vxZ[vecinomxpymz].y +
    dlxm * dlyp * dlz  * vxZ[vecinomxpy].y +
    dlxm * dlyp * dlzp * vxZ[vecinomxpypz].y +
    dlx  * dlym * dlzm * vxZ[vecinomymz].y +
    dlx  * dlym * dlz  * vxZ[vecino1].y +
    dlx  * dlym * dlzp * vxZ[vecinomypz].y + 
    dlx  * dly  * dlzm * vxZ[vecino0].y + 
    dlx  * dly  * dlz  * vxZ[icelx].y + 
    dlx  * dly  * dlzp * vxZ[vecino5].y + 
    dlx  * dlyp * dlzm * vxZ[vecinopymz].y + 
    dlx  * dlyp * dlz  * vxZ[vecino4].y + 
    dlx  * dlyp * dlzp * vxZ[vecinopypz].y + 
    dlxp * dlym * dlzm * vxZ[vecinopxmymz].y + 
    dlxp * dlym * dlz  * vxZ[vecinopxmy].y + 
    dlxp * dlym * dlzp * vxZ[vecinopxmypz].y +
    dlxp * dly  * dlzm * vxZ[vecinopxmz].y + 
    dlxp * dly  * dlz  * vxZ[vecino3].y +
    dlxp * dly  * dlzp * vxZ[vecinopxpz].y +
    dlxp * dlyp * dlzm * vxZ[vecinopxpymz].y +
    dlxp * dlyp * dlz  * vxZ[vecinopxpy].y +
    dlxp * dlyp * dlzp * vxZ[vecinopxpypz].y;


  
  double SDeltap = (vxboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU);
  


  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlxp * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlxp * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlxp * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx  * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx  * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx  * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlxm * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = dlxm * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = dlxm * dlym * dlzm * SDeltap;





  //SPREAD IN THE Y DIRECTION
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

  tildev = dlxm * dlym * dlzm * vyZ[vecinomxmymz].y +
    dlxm * dlym * dlz  * vyZ[vecinomxmy].y +
    dlxm * dlym * dlzp * vyZ[vecinomxmypz].y +
    dlxm * dly  * dlzm * vyZ[vecinomxmz].y + 
    dlxm * dly  * dlz  * vyZ[vecino2].y +
    dlxm * dly  * dlzp * vyZ[vecinomxpz].y +
    dlxm * dlyp * dlzm * vyZ[vecinomxpymz].y +
    dlxm * dlyp * dlz  * vyZ[vecinomxpy].y +
    dlxm * dlyp * dlzp * vyZ[vecinomxpypz].y +
    dlx  * dlym * dlzm * vyZ[vecinomymz].y +
    dlx  * dlym * dlz  * vyZ[vecino1].y +
    dlx  * dlym * dlzp * vyZ[vecinomypz].y + 
    dlx  * dly  * dlzm * vyZ[vecino0].y + 
    dlx  * dly  * dlz  * vyZ[icely].y + 
    dlx  * dly  * dlzp * vyZ[vecino5].y + 
    dlx  * dlyp * dlzm * vyZ[vecinopymz].y + 
    dlx  * dlyp * dlz  * vyZ[vecino4].y + 
    dlx  * dlyp * dlzp * vyZ[vecinopypz].y + 
    dlxp * dlym * dlzm * vyZ[vecinopxmymz].y + 
    dlxp * dlym * dlz  * vyZ[vecinopxmy].y + 
    dlxp * dlym * dlzp * vyZ[vecinopxmypz].y +
    dlxp * dly  * dlzm * vyZ[vecinopxmz].y + 
    dlxp * dly  * dlz  * vyZ[vecino3].y +
    dlxp * dly  * dlzp * vyZ[vecinopxpz].y +
    dlxp * dlyp * dlzm * vyZ[vecinopxpymz].y +
    dlxp * dlyp * dlz  * vyZ[vecinopxpy].y +
    dlxp * dlyp * dlzp * vyZ[vecinopxpypz].y; 


  SDeltap = (vyboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU) ;
  
  


  offset = nboundaryGPU;
  fyboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlxp * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlxp * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlxp * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx  * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx  * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx  * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlxm * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = dlxm * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = dlxm * dlym * dlzm * SDeltap;





  //SPREAD IN THE Z DIRECTION
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


  tildev = dlxm * dlym * dlzm * vzZ[vecinomxmymz].y +
    dlxm * dlym * dlz  * vzZ[vecinomxmy].y +
    dlxm * dlym * dlzp * vzZ[vecinomxmypz].y +
    dlxm * dly  * dlzm * vzZ[vecinomxmz].y + 
    dlxm * dly  * dlz  * vzZ[vecino2].y +
    dlxm * dly  * dlzp * vzZ[vecinomxpz].y +
    dlxm * dlyp * dlzm * vzZ[vecinomxpymz].y +
    dlxm * dlyp * dlz  * vzZ[vecinomxpy].y +
    dlxm * dlyp * dlzp * vzZ[vecinomxpypz].y +
    dlx  * dlym * dlzm * vzZ[vecinomymz].y +
    dlx  * dlym * dlz  * vzZ[vecino1].y +
    dlx  * dlym * dlzp * vzZ[vecinomypz].y + 
    dlx  * dly  * dlzm * vzZ[vecino0].y + 
    dlx  * dly  * dlz  * vzZ[icelz].y + 
    dlx  * dly  * dlzp * vzZ[vecino5].y + 
    dlx  * dlyp * dlzm * vzZ[vecinopymz].y + 
    dlx  * dlyp * dlz  * vzZ[vecino4].y + 
    dlx  * dlyp * dlzp * vzZ[vecinopypz].y + 
    dlxp * dlym * dlzm * vzZ[vecinopxmymz].y + 
    dlxp * dlym * dlz  * vzZ[vecinopxmy].y + 
    dlxp * dlym * dlzp * vzZ[vecinopxmypz].y +
    dlxp * dly  * dlzm * vzZ[vecinopxmz].y + 
    dlxp * dly  * dlz  * vzZ[vecino3].y +
    dlxp * dly  * dlzp * vzZ[vecinopxpz].y +
    dlxp * dlyp * dlzm * vzZ[vecinopxpymz].y +
    dlxp * dlyp * dlz  * vzZ[vecinopxpy].y +
    dlxp * dlyp * dlzp * vzZ[vecinopxpypz].y;


  SDeltap = (vzboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU);
  

  

  offset = nboundaryGPU;
  fzboundaryGPU[offset+i]        = dlxp * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = dlxp * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = dlxp * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = dlxp * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = dlxp * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = dlxp * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = dlx  * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = dlx  * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = dlx  * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = dlx  * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = dlx  * dlym * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = dlxm * dlyp * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = dlxm * dly  * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = dlxm * dly  * dlzm * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzp * SDeltap;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = dlxm * dlym * dlz  * SDeltap;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = dlxm * dlym * dlzm * SDeltap;





}


























//Update particle velocity if me=0
__global__ void updateParticleVelocityme0(double* rxcellGPU,
					  double* rycellGPU,
					  double* rzcellGPU,
					  double* vxboundaryGPU,
					  double* vyboundaryGPU,
					  double* vzboundaryGPU){
					  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  double r, rp, rm;

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

 
  double v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);


  vxboundaryGPU[i] = v;


  //SPREAD IN THE Y DIRECTION
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

  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz); 


  vyboundaryGPU[i] = v;
  

  //SPREAD IN THE Z DIRECTION
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


  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);


  vzboundaryGPU[i] = v;
  

}



















//Update particle velocity if me!=0
__global__ void updateParticleVelocityme(double* rxcellGPU,
					 double* rycellGPU,
					 double* rzcellGPU,
					 cufftDoubleComplex* vxZ,
					 cufftDoubleComplex* vyZ,
					 cufftDoubleComplex* vzZ,
					 double* vxboundaryGPU,
					 double* vyboundaryGPU,
					 double* vzboundaryGPU,
					 double* vxboundaryPredictionGPU,
					 double* vyboundaryPredictionGPU,
					 double* vzboundaryPredictionGPU){
					  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  double r, rp, rm;

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

 
  double v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);

  double tildev = dlxm * dlym * dlzm * vxZ[vecinomxmymz].y +
    dlxm * dlym * dlz  * vxZ[vecinomxmy].y +
    dlxm * dlym * dlzp * vxZ[vecinomxmypz].y +
    dlxm * dly  * dlzm * vxZ[vecinomxmz].y + 
    dlxm * dly  * dlz  * vxZ[vecino2].y +
    dlxm * dly  * dlzp * vxZ[vecinomxpz].y +
    dlxm * dlyp * dlzm * vxZ[vecinomxpymz].y +
    dlxm * dlyp * dlz  * vxZ[vecinomxpy].y +
    dlxm * dlyp * dlzp * vxZ[vecinomxpypz].y +
    dlx  * dlym * dlzm * vxZ[vecinomymz].y +
    dlx  * dlym * dlz  * vxZ[vecino1].y +
    dlx  * dlym * dlzp * vxZ[vecinomypz].y + 
    dlx  * dly  * dlzm * vxZ[vecino0].y + 
    dlx  * dly  * dlz  * vxZ[icelx].y + 
    dlx  * dly  * dlzp * vxZ[vecino5].y + 
    dlx  * dlyp * dlzm * vxZ[vecinopymz].y + 
    dlx  * dlyp * dlz  * vxZ[vecino4].y + 
    dlx  * dlyp * dlzp * vxZ[vecinopypz].y + 
    dlxp * dlym * dlzm * vxZ[vecinopxmymz].y + 
    dlxp * dlym * dlz  * vxZ[vecinopxmy].y + 
    dlxp * dlym * dlzp * vxZ[vecinopxmypz].y +
    dlxp * dly  * dlzm * vxZ[vecinopxmz].y + 
    dlxp * dly  * dlz  * vxZ[vecino3].y +
    dlxp * dly  * dlzp * vxZ[vecinopxpz].y +
    dlxp * dlyp * dlzm * vxZ[vecinopxpymz].y +
    dlxp * dlyp * dlz  * vxZ[vecinopxpy].y +
    dlxp * dlyp * dlzp * vxZ[vecinopxpypz].y;



  vxboundaryGPU[i] = v + tildev + vxboundaryPredictionGPU[i];


  //SPREAD IN THE Y DIRECTION
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

  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz); 


  tildev = dlxm * dlym * dlzm * vyZ[vecinomxmymz].y +
    dlxm * dlym * dlz  * vyZ[vecinomxmy].y +
    dlxm * dlym * dlzp * vyZ[vecinomxmypz].y +
    dlxm * dly  * dlzm * vyZ[vecinomxmz].y + 
    dlxm * dly  * dlz  * vyZ[vecino2].y +
    dlxm * dly  * dlzp * vyZ[vecinomxpz].y +
    dlxm * dlyp * dlzm * vyZ[vecinomxpymz].y +
    dlxm * dlyp * dlz  * vyZ[vecinomxpy].y +
    dlxm * dlyp * dlzp * vyZ[vecinomxpypz].y +
    dlx  * dlym * dlzm * vyZ[vecinomymz].y +
    dlx  * dlym * dlz  * vyZ[vecino1].y +
    dlx  * dlym * dlzp * vyZ[vecinomypz].y + 
    dlx  * dly  * dlzm * vyZ[vecino0].y + 
    dlx  * dly  * dlz  * vyZ[icely].y + 
    dlx  * dly  * dlzp * vyZ[vecino5].y + 
    dlx  * dlyp * dlzm * vyZ[vecinopymz].y + 
    dlx  * dlyp * dlz  * vyZ[vecino4].y + 
    dlx  * dlyp * dlzp * vyZ[vecinopypz].y + 
    dlxp * dlym * dlzm * vyZ[vecinopxmymz].y + 
    dlxp * dlym * dlz  * vyZ[vecinopxmy].y + 
    dlxp * dlym * dlzp * vyZ[vecinopxmypz].y +
    dlxp * dly  * dlzm * vyZ[vecinopxmz].y + 
    dlxp * dly  * dlz  * vyZ[vecino3].y +
    dlxp * dly  * dlzp * vyZ[vecinopxpz].y +
    dlxp * dlyp * dlzm * vyZ[vecinopxpymz].y +
    dlxp * dlyp * dlz  * vyZ[vecinopxpy].y +
    dlxp * dlyp * dlzp * vyZ[vecinopxpypz].y; 


  vyboundaryGPU[i] = v + tildev + vyboundaryPredictionGPU[i];

  

  //SPREAD IN THE Z DIRECTION
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


  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);


  tildev = dlxm * dlym * dlzm * vzZ[vecinomxmymz].y +
    dlxm * dlym * dlz  * vzZ[vecinomxmy].y +
    dlxm * dlym * dlzp * vzZ[vecinomxmypz].y +
    dlxm * dly  * dlzm * vzZ[vecinomxmz].y + 
    dlxm * dly  * dlz  * vzZ[vecino2].y +
    dlxm * dly  * dlzp * vzZ[vecinomxpz].y +
    dlxm * dlyp * dlzm * vzZ[vecinomxpymz].y +
    dlxm * dlyp * dlz  * vzZ[vecinomxpy].y +
    dlxm * dlyp * dlzp * vzZ[vecinomxpypz].y +
    dlx  * dlym * dlzm * vzZ[vecinomymz].y +
    dlx  * dlym * dlz  * vzZ[vecino1].y +
    dlx  * dlym * dlzp * vzZ[vecinomypz].y + 
    dlx  * dly  * dlzm * vzZ[vecino0].y + 
    dlx  * dly  * dlz  * vzZ[icelz].y + 
    dlx  * dly  * dlzp * vzZ[vecino5].y + 
    dlx  * dlyp * dlzm * vzZ[vecinopymz].y + 
    dlx  * dlyp * dlz  * vzZ[vecino4].y + 
    dlx  * dlyp * dlzp * vzZ[vecinopypz].y + 
    dlxp * dlym * dlzm * vzZ[vecinopxmymz].y + 
    dlxp * dlym * dlz  * vzZ[vecinopxmy].y + 
    dlxp * dlym * dlzp * vzZ[vecinopxmypz].y +
    dlxp * dly  * dlzm * vzZ[vecinopxmz].y + 
    dlxp * dly  * dlz  * vzZ[vecino3].y +
    dlxp * dly  * dlzp * vzZ[vecinopxpz].y +
    dlxp * dlyp * dlzm * vzZ[vecinopxpymz].y +
    dlxp * dlyp * dlz  * vzZ[vecinopxpy].y +
    dlxp * dlyp * dlzp * vzZ[vecinopxpypz].y;


  vzboundaryGPU[i] = v + tildev + vzboundaryPredictionGPU[i];

  
}









__global__ void updateFluidme0(double* vxGPU,
			       double* vyGPU,
			       double* vzGPU,
			       double* vxPredictionGPU,
			       double* vyPredictionGPU,
			       double* vzPredictionGPU){			       

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   


  vxGPU[i] = vxPredictionGPU[i] ;
  vyGPU[i] = vyPredictionGPU[i] ;
  vzGPU[i] = vzPredictionGPU[i] ;

}
































//FOR STEP 5, CALCULATE \Delta p
//Calculate \Delta p and saved in vxboundaryGPU
__global__ void kernelCalculateDeltapFirstStep(double* rxcellGPU,
					       double* rycellGPU,
					       double* rzcellGPU,
					       double* vxboundaryGPU,
					       double* vyboundaryGPU,
					       double* vzboundaryGPU,
					       double* vxboundaryPredictionGPU,
					       double* vyboundaryPredictionGPU,
					       double* vzboundaryPredictionGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   


  //Particle position at time n+1/2
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);

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





  //X DIRECTION
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


  double v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);

  
  vxboundaryPredictionGPU[i] = massParticleGPU * ( vxboundaryGPU[i] - v - vxboundaryPredictionGPU[i] );


  //Y DIRECTION
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


  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz); 

  
  vyboundaryPredictionGPU[i] = massParticleGPU * ( vyboundaryGPU[i] - v - vyboundaryPredictionGPU[i] );


  //Z DIRECTION
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

  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);

  vzboundaryPredictionGPU[i] = massParticleGPU * ( vzboundaryGPU[i] - v - vzboundaryPredictionGPU[i] );
  

}

























__global__ void calculateVelocityAtHalfTimeStepme0(double* vxGPU,
						   double* vyGPU,
						   double* vzGPU,
						   double* vxPredictionGPU,
						   double* vyPredictionGPU,
						   double* vzPredictionGPU){
						

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  vxGPU[i] = 0.5 * (vxGPU[i] + vxPredictionGPU[i] );
  vyGPU[i] = 0.5 * (vyGPU[i] + vyPredictionGPU[i] );
  vzGPU[i] = 0.5 * (vzGPU[i] + vzPredictionGPU[i] );

}
