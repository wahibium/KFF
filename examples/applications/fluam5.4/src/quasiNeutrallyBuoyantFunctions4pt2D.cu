__global__ void findNeighborParticlesQuasiNeutrallyBuoyant4pt_1_2D(particlesincell* pc, 
								   int* errorKernel,
								   const double* rxcellGPU,
								   const double* rycellGPU,
								   const double* rzcellGPU,
								   const double* rxboundaryGPU,  //q^{n}
								   const double* ryboundaryGPU, 
								   const double* rzboundaryGPU,
								   double* rxboundaryPredictionGPU, //q^{n+1/2}
								   double* ryboundaryPredictionGPU, 
								   double* rzboundaryPredictionGPU,
								   const double* vxGPU, //v^n
								   const double* vyGPU, 
								   const double* vzGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  
  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;//Primitive image

  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;

  int icel;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel = jx + jy*mxGPU;
 
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  

  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * vxGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vxGPU[vecinomxmxmy] +
    dlx0 * dly2 * vxGPU[vecinomxmx] +
    dlx0 * dly3 * vxGPU[vecinomxmxpy] +
    dlx0 * dly4 * vxGPU[vecinomxmxpypy] +
    dlx1 * dly0 * vxGPU[vecinomxmymy] +
    dlx1 * dly1 * vxGPU[vecinomxmy] +
    dlx1 * dly2 * vxGPU[vecino2] +
    dlx1 * dly3 * vxGPU[vecinomxpy] +
    dlx1 * dly4 * vxGPU[vecinomxpypy] +
    dlx2 * dly0 * vxGPU[vecinomymy] +
    dlx2 * dly1 * vxGPU[vecino1] +
    dlx2 * dly2 * vxGPU[icel] +
    dlx2 * dly3 * vxGPU[vecino4] +
    dlx2 * dly4 * vxGPU[vecinopypy] +
    dlx3 * dly0 * vxGPU[vecinopxmymy] +
    dlx3 * dly1 * vxGPU[vecinopxmy] +
    dlx3 * dly2 * vxGPU[vecino3] +
    dlx3 * dly3 * vxGPU[vecinopxpy] +
    dlx3 * dly4 * vxGPU[vecinopxpypy];
  
  
  double rxNew = rx + 0.5 * dtGPU * v;
  rxboundaryPredictionGPU[nboundaryGPU+i] = rxNew;


  //VELOCITY IN THE Y DIRECTION
  //vecino1 = tex1Dfetch(texvecino1GPU, icel);
  //vecino2 = tex1Dfetch(texvecino2GPU, icel);
  //vecino3 = tex1Dfetch(texvecino3GPU, icel);
  //vecino4 = tex1Dfetch(texvecino4GPU, icel);
  //vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  //vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  //vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  //vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  //DEFINE MORE NEIGHBORS
  //vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  //vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  //vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  //vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  //vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  //vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);
  //vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);



  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);


  v = dlx0 * dly0 * vyGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vyGPU[vecinomxmxmy] +
    dlx0 * dly2 * vyGPU[vecinomxmx] +
    dlx0 * dly3 * vyGPU[vecinomxmxpy] +
    dlx1 * dly0 * vyGPU[vecinomxmymy] +
    dlx1 * dly1 * vyGPU[vecinomxmy] +
    dlx1 * dly2 * vyGPU[vecino2] +
    dlx1 * dly3 * vyGPU[vecinomxpy] +
    dlx2 * dly0 * vyGPU[vecinomymy] +
    dlx2 * dly1 * vyGPU[vecino1] +
    dlx2 * dly2 * vyGPU[icel] +
    dlx2 * dly3 * vyGPU[vecino4] +
    dlx3 * dly0 * vyGPU[vecinopxmymy] +
    dlx3 * dly1 * vyGPU[vecinopxmy] +
    dlx3 * dly2 * vyGPU[vecino3] +
    dlx3 * dly3 * vyGPU[vecinopxpy] +
    dlx4 * dly0 * vyGPU[vecinopxpxmymy] +
    dlx4 * dly1 * vyGPU[vecinopxpxmy] +
    dlx4 * dly2 * vyGPU[vecinopxpx] +
    dlx4 * dly3 * vyGPU[vecinopxpxpy];
  

  double ryNew = ry + 0.5 * dtGPU * v;

  ryboundaryPredictionGPU[nboundaryGPU+i] = ryNew;
 
  
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    rxNew = rxNew - (int(rxNew*invlxGPU + 0.5*((rxNew>0)-(rxNew<0)))) * lxGPU;
    int jx   = int(rxNew * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    ryNew = ryNew - (int(ryNew*invlyGPU + 0.5*((ryNew>0)-(ryNew<0)))) * lyGPU;
    int jy   = int(ryNew * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
  }
  int np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;

}



























//STEP 2: CALCULATE FORCES AND SPREAD THEM TO THE FLUID S^{n+1/2} * F^{n+1/2}
//Fill "countparticlesincellX" lists
//and spread particle force F 
//Fill "countparticlesincellX" lists
//and spread particle force F 
__global__ void kernelSpreadParticlesForce4pt2D(const double* rxcellGPU, 
						const double* rycellGPU, 
						const double* rzcellGPU,
						double* fxboundaryGPU, 
						double* fyboundaryGPU, 
						double* fzboundaryGPU,
						const particlesincell* pc, 
						int* errorKernel,
						const bondedForcesVariables* bFV){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double fx = 0.;//pressurea0GPU;// * 0.267261241912424385 ;//0;
  double fy = 0.;//pressurea0GPU * 0.534522483824848769 ;
  double f;
 

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  
  //INCLUDE EXTERNAL FORCES HERE

  //Example: harmonic potential 
  // V(r) = (1/2) * k * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
  //
  //with spring constant k=0.01
  //and x0=y0=z0=0
  //
  //fx = -0.01*rx;
  //fy = -0.01*ry;
  //fz = -0.01*rz;




  //NEW bonded forces
  if(bondedForcesGPU){
    //call function for bonded forces particle-particle
    forceBondedParticleParticleGPU_2D(i,
				      fx,
				      fy,
				      rx,
				      ry,
				      bFV);
  }
    
  double rxij, ryij, r2;

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;
  
  int icel;
  double r;
  double rxI, ryI;

  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
  }
  
  int np;
  if(computeNonBondedForcesGPU){
    //Particles in Cell i
    np = tex1Dfetch(texCountParticlesInCellNonBonded,icel);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+icel);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }  
    //Particles in Cell vecino1
    vecino1 = tex1Dfetch(texneighbor1GPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino1);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino1);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecino2
    vecino2 = tex1Dfetch(texneighbor2GPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino2);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino2);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecino3
    vecino3 = tex1Dfetch(texneighbor3GPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino3);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino3);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecino4
    vecino4 = tex1Dfetch(texneighbor4GPU, icel);
    //printf("VECINO %i %i \n",icel,vecino4);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino4);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino4);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinopxpy
    vecinopxpy = tex1Dfetch(texneighborpxpyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpy);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinopxmy
    vecinopxmy = tex1Dfetch(texneighborpxmyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmy);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinomxpy
    vecinomxpy = tex1Dfetch(texneighbormxpyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpy);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinomxmy
    vecinomxmy = tex1Dfetch(texneighbormxmyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmy);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
  }


  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel = jx + jy*mxGPU;
  }

  np = atomicAdd(&pc->countparticlesincellX[icel],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellstGPU*np+icel] = nboundaryGPU+i;


  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;

  //FORCE IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4 = 0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

    
  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = -dlx4 * dly4 * fx;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = -dlx4 * dly3 * fx;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = -dlx4 * dly2 * fx;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = -dlx4 * dly1 * fx;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = -dlx4 * dly0 * fx;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = -dlx3 * dly4 * fx;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = -dlx3 * dly3 * fx;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = -dlx3 * dly2 * fx;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = -dlx3 * dly1 * fx;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = -dlx3 * dly0 * fx;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = -dlx2 * dly4 * fx;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = -dlx2 * dly3 * fx;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = -dlx2 * dly2 * fx;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = -dlx2 * dly1 * fx;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = -dlx2 * dly0 * fx;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = -dlx1 * dly4 * fx;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = -dlx1 * dly3 * fx;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = -dlx1 * dly2 * fx;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = -dlx1 * dly1 * fx;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = -dlx1 * dly0 * fx;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = -dlx0 * dly4 * fx;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = -dlx0 * dly3 * fx;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = -dlx0 * dly2 * fx;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = -dlx0 * dly1 * fx;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = -dlx0 * dly0 * fx;


  
  
  //FORCE IN THE Y DIRECTION
  //vecino1 = tex1Dfetch(texvecino1GPU, icel);
  //vecino2 = tex1Dfetch(texvecino2GPU, icel);
  //vecino3 = tex1Dfetch(texvecino3GPU, icel);
  //vecino4 = tex1Dfetch(texvecino4GPU, icel);
  //vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  //vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  //vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  //vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  //DEFINE MORE NEIGHBORS
  //vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  //vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  //vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  //vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  //vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  //vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);
  //vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;


  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = -dlx4 * dly4 * fy;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = -dlx4 * dly3 * fy;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = -dlx4 * dly2 * fy;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = -dlx4 * dly1 * fy;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = -dlx4 * dly0 * fy;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = -dlx3 * dly4 * fy;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = -dlx3 * dly3 * fy;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = -dlx3 * dly2 * fy;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = -dlx3 * dly1 * fy;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = -dlx3 * dly0 * fy;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = -dlx2 * dly4 * fy;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = -dlx2 * dly3 * fy;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = -dlx2 * dly2 * fy;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = -dlx2 * dly1 * fy;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = -dlx2 * dly0 * fy;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = -dlx1 * dly4 * fy;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = -dlx1 * dly3 * fy;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = -dlx1 * dly2 * fy;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = -dlx1 * dly1 * fy;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = -dlx1 * dly0 * fy;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = -dlx0 * dly4 * fy;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = -dlx0 * dly3 * fy;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = -dlx0 * dly2 * fy;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = -dlx0 * dly1 * fy;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = -dlx0 * dly0 * fy;

  
}

























//In this kernel we construct the vector W
//
// W = v^n + 0.5*dt*nu*L*v^n + Advection^{n+1/2} + (dt/rho)*f^n_{noise} + dt*SF/rho 
//
__global__ void kernelConstructWQuasiNeutrallyBuoyant4pt_2D(const double *vxPredictionGPU, 
							    const double *vyPredictionGPU, 
							    const double *vzPredictionGPU, 
							    cufftDoubleComplex *WxZ, 
							    cufftDoubleComplex *WyZ, 
							    cufftDoubleComplex *WzZ, 
							    const double *d_rand,
							    const double *fxboundaryGPU,
							    const double *fyboundaryGPU,
							    const double *fzboundaryGPU,
							    double* advXGPU,
							    double* advYGPU,
							    double* advZGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy;
  double vx, vy;
  double vx1, vx2, vx3, vx4;
  double vy1, vy2, vy3, vy4;
  int vecino1, vecino2, vecino3, vecino4; 
  int vecinopxmy;
  int vecinomxpy;
  double vxmxpy;
  double vypxmy;

  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);



  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;

  //Previous Velocity
  wx += vx ;//- pressurea1GPU * dtGPU;// * 0.267261241912424385 * dtGPU;
  wy += vy ;//- pressurea1GPU * 0.534522483824848769 * dtGPU;
  
  //Advection part
  double advX, advY;
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY  = 0.25 * dtGPU * advY;
  

  //NOISE part
  double dnoise_sXX, dnoise_sXY;
  double dnoise_sYY;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 2*ncellsGPU] ;
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/2.;
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 2*ncellsGPU] ;
  dnoise_sYY = d_rand[vecino4 + 2*ncellsGPU] - dnoise_tr/2.;
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4GPU * dnoise_sXY;
  wy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_tr = d_rand[i] + d_rand[i + 2*ncellsGPU] ;
  dnoise_sXX = d_rand[i] - dnoise_tr/2.;
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 2*ncellsGPU] - dnoise_tr/2.;
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4GPU * dnoise_sXY;


  int vecino, particle;
  double fx = 0.f;
  double fy = 0.f;


  //Particle contribution
  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  np = tex1Dfetch(texCountParticlesInCellX,vecino1);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino1);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  np = tex1Dfetch(texCountParticlesInCellX,vecino2);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino2);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];    
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  np = tex1Dfetch(texCountParticlesInCellX,vecino3);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino3);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  np = tex1Dfetch(texCountParticlesInCellX,vecino4);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino4);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  int vecinopxpy = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxpy);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxmy);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxpy);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  int vecinomxmy = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxmy);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymy
  vecino = tex1Dfetch(texvecino1GPU, vecino1);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxmymy
  vecino = tex1Dfetch(texvecinomxmyGPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle ];
    fy -= fyboundaryGPU[particle ];
  }
  //Particles in Cell vecinomxmxmy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+ (nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+ (nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmx
  vecino = tex1Dfetch(texvecino2GPU, vecino2);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpypy
  vecino = tex1Dfetch(texvecinomxpyGPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypy
  vecino = tex1Dfetch(texvecino4GPU, vecino4);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpypy
  vecino = tex1Dfetch(texvecinopxpyGPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpx
  vecino = tex1Dfetch(texvecino3GPU, vecino3);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmymy
  vecino = tex1Dfetch(texvecinopxmyGPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }








  WxZ[i].x = wx + dtGPU * ( fx / (volumeGPU*densfluidGPU) ) - advX  ;
  WyZ[i].x = wy + dtGPU * ( fy / (volumeGPU*densfluidGPU) ) - advY  ;
  WzZ[i].x = 0;//wz + dtGPU * ( fz / (volumeGPU*densfluidGPU) ) - advZ  ;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

  //Save advection for the next time step
  advXGPU[i] = advX ;
  advYGPU[i] = advY ;

}



























//FOR STEP 4
//Calculate \delta u^{n+1/2} and saved in vxboundaryPredictionGPU
__global__ void kernelCalculateDeltau4pt_2D(const double* rxcellGPU,
					    const double* rycellGPU,
					    const double* rzcellGPU,
					    const double* vxGPU,
					    const double* vyGPU,
					    const double* vzGPU,
					    const double* rxboundaryGPU,
					    const double* ryboundaryGPU,
					    const double* rzboundaryGPU,
					    double* vxboundaryPredictionGPU,
					    double* vyboundaryPredictionGPU,
					    double* vzboundaryPredictionGPU){

 
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  //delta u^{n+1/2}
  double dux;
  double duy;

  //Particle position at time n+1/2
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  int icel;


  //FIRST CALCULATE J^{n+1/2} * v^n
  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy * mxGPU;
  }





  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * vxGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vxGPU[vecinomxmxmy] +
    dlx0 * dly2 * vxGPU[vecinomxmx] +
    dlx0 * dly3 * vxGPU[vecinomxmxpy] +
    dlx0 * dly4 * vxGPU[vecinomxmxpypy] +
    dlx1 * dly0 * vxGPU[vecinomxmymy] +
    dlx1 * dly1 * vxGPU[vecinomxmy] +
    dlx1 * dly2 * vxGPU[vecino2] +
    dlx1 * dly3 * vxGPU[vecinomxpy] +
    dlx1 * dly4 * vxGPU[vecinomxpypy] +
    dlx2 * dly0 * vxGPU[vecinomymy] +
    dlx2 * dly1 * vxGPU[vecino1] +
    dlx2 * dly2 * vxGPU[icel] +
    dlx2 * dly3 * vxGPU[vecino4] +
    dlx2 * dly4 * vxGPU[vecinopypy] +
    dlx3 * dly0 * vxGPU[vecinopxmymy] +
    dlx3 * dly1 * vxGPU[vecinopxmy] +
    dlx3 * dly2 * vxGPU[vecino3] +
    dlx3 * dly3 * vxGPU[vecinopxpy] +
    dlx3 * dly4 * vxGPU[vecinopxpypy];


  dux = v;


  //Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);


  v = dlx0 * dly0 * vyGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vyGPU[vecinomxmxmy] +
    dlx0 * dly2 * vyGPU[vecinomxmx] +
    dlx0 * dly3 * vyGPU[vecinomxmxpy] +
    dlx1 * dly0 * vyGPU[vecinomxmymy] +
    dlx1 * dly1 * vyGPU[vecinomxmy] +
    dlx1 * dly2 * vyGPU[vecino2] +
    dlx1 * dly3 * vyGPU[vecinomxpy] +
    dlx2 * dly0 * vyGPU[vecinomymy] +
    dlx2 * dly1 * vyGPU[vecino1] +
    dlx2 * dly2 * vyGPU[icel] +
    dlx2 * dly3 * vyGPU[vecino4] +
    dlx3 * dly0 * vyGPU[vecinopxmymy] +
    dlx3 * dly1 * vyGPU[vecinopxmy] +
    dlx3 * dly2 * vyGPU[vecino3] +
    dlx3 * dly3 * vyGPU[vecinopxpy] +
    dlx4 * dly0 * vyGPU[vecinopxpxmymy] +
    dlx4 * dly1 * vyGPU[vecinopxpxmy] +
    dlx4 * dly2 * vyGPU[vecinopxpx] +
    dlx4 * dly3 * vyGPU[vecinopxpxpy];
    
  duy = v;




  //SECOND CALCULATE J^{n} * v^n

  //Particle positon at time n
  rx = rxboundaryGPU[i];
  ry = ryboundaryGPU[i];


  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }





  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  v = dlx0 * dly0 * vxGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vxGPU[vecinomxmxmy] +
    dlx0 * dly2 * vxGPU[vecinomxmx] +
    dlx0 * dly3 * vxGPU[vecinomxmxpy] +
    dlx0 * dly4 * vxGPU[vecinomxmxpypy] +
    dlx1 * dly0 * vxGPU[vecinomxmymy] +
    dlx1 * dly1 * vxGPU[vecinomxmy] +
    dlx1 * dly2 * vxGPU[vecino2] +
    dlx1 * dly3 * vxGPU[vecinomxpy] +
    dlx1 * dly4 * vxGPU[vecinomxpypy] +
    dlx2 * dly0 * vxGPU[vecinomymy] +
    dlx2 * dly1 * vxGPU[vecino1] +
    dlx2 * dly2 * vxGPU[icel] +
    dlx2 * dly3 * vxGPU[vecino4] +
    dlx2 * dly4 * vxGPU[vecinopypy] +
    dlx3 * dly0 * vxGPU[vecinopxmymy] +
    dlx3 * dly1 * vxGPU[vecinopxmy] +
    dlx3 * dly2 * vxGPU[vecino3] +
    dlx3 * dly3 * vxGPU[vecinopxpy] +
    dlx3 * dly4 * vxGPU[vecinopxpypy];

  dux -= v;


  //Y DIRECTION
  vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);


  v = dlx0 * dly0 * vyGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vyGPU[vecinomxmxmy] +
    dlx0 * dly2 * vyGPU[vecinomxmx] +
    dlx0 * dly3 * vyGPU[vecinomxmxpy] +
    dlx1 * dly0 * vyGPU[vecinomxmymy] +
    dlx1 * dly1 * vyGPU[vecinomxmy] +
    dlx1 * dly2 * vyGPU[vecino2] +
    dlx1 * dly3 * vyGPU[vecinomxpy] +
    dlx2 * dly0 * vyGPU[vecinomymy] +
    dlx2 * dly1 * vyGPU[vecino1] +
    dlx2 * dly2 * vyGPU[icel] +
    dlx2 * dly3 * vyGPU[vecino4] +
    dlx3 * dly0 * vyGPU[vecinopxmymy] +
    dlx3 * dly1 * vyGPU[vecinopxmy] +
    dlx3 * dly2 * vyGPU[vecino3] +
    dlx3 * dly3 * vyGPU[vecinopxpy] +
    dlx4 * dly0 * vyGPU[vecinopxpxmymy] +
    dlx4 * dly1 * vyGPU[vecinopxpxmy] +
    dlx4 * dly2 * vyGPU[vecinopxpx] +
    dlx4 * dly3 * vyGPU[vecinopxpxpy];
    
  duy -= v;


  //THIRD, add (J^{n+1/2} - J^n) * v^n to
  //0.5*dt*nu * J^{n-1/2} * L * \Delta v^{n+1/2}
  //vxboundaryPredictionGPU[i] += dux;
  //vyboundaryPredictionGPU[i] += duy;

  //\delta u = J*L*Delta u
  //vxboundaryPredictionGPU[i] += 0;
  //vyboundaryPredictionGPU[i] += 0;

  //\delta u = 0
  vxboundaryPredictionGPU[i] = 0;
  vyboundaryPredictionGPU[i] = 0;


}

























//FOR STEP 5, CALCULATE \Delta p
//Calculate \Delta p and saved in vxboundaryPredictionGPU
__global__ void kernelCalculateDeltapFirstStep4pt_2D(const double* rxcellGPU,
						     const double* rycellGPU,
						     const double* rzcellGPU,
						     const double* vxboundaryGPU,
						     const double* vyboundaryGPU,
						     const double* vzboundaryGPU,
						     double* vxboundaryPredictionGPU,
						     double* vyboundaryPredictionGPU,
						     double* vzboundaryPredictionGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   


  //Particle position at time n+1/2
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  int icel;


  
  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }


  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * fetch_double(texVxGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVxGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVxGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVxGPU,vecinomxmxpy) +
    dlx0 * dly4 * fetch_double(texVxGPU,vecinomxmxpypy) +
    dlx1 * dly0 * fetch_double(texVxGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVxGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVxGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVxGPU,vecinomxpy) +
    dlx1 * dly4 * fetch_double(texVxGPU,vecinomxpypy) +
    dlx2 * dly0 * fetch_double(texVxGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVxGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVxGPU,icel) +
    dlx2 * dly3 * fetch_double(texVxGPU,vecino4) +
    dlx2 * dly4 * fetch_double(texVxGPU,vecinopypy) +
    dlx3 * dly0 * fetch_double(texVxGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVxGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVxGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVxGPU,vecinopxpy) +
    dlx3 * dly4 * fetch_double(texVxGPU,vecinopxpypy);

  
  vxboundaryPredictionGPU[i] = massParticleGPU * ( vxboundaryGPU[i] - v - vxboundaryPredictionGPU[i] );


  //Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);


  v = dlx0 * dly0 * fetch_double(texVyGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVyGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVyGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVyGPU,vecinomxmxpy) +
    dlx1 * dly0 * fetch_double(texVyGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVyGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVyGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVyGPU,vecinomxpy) +
    dlx2 * dly0 * fetch_double(texVyGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVyGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVyGPU,icel) +
    dlx2 * dly3 * fetch_double(texVyGPU,vecino4) +
    dlx3 * dly0 * fetch_double(texVyGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVyGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVyGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVyGPU,vecinopxpy) +
    dlx4 * dly0 * fetch_double(texVyGPU,vecinopxpxmymy) +
    dlx4 * dly1 * fetch_double(texVyGPU,vecinopxpxmy) +
    dlx4 * dly2 * fetch_double(texVyGPU,vecinopxpx) +
    dlx4 * dly3 * fetch_double(texVyGPU,vecinopxpxpy);

  
  vyboundaryPredictionGPU[i] = massParticleGPU * ( vyboundaryGPU[i] - v - vyboundaryPredictionGPU[i] );


}


















//First, spread \Delta p, prefactor * S*{\Delta p}
__global__ void kernelSpreadDeltap4pt2D(const double* rxcellGPU,
					const double* rycellGPU,
					const double* rzcellGPU,
					const double* vxboundaryGPU,
					const double* vyboundaryGPU,
					const double* vzboundaryGPU,
					double* fxboundaryGPU,
					double* fyboundaryGPU,
					double* fzboundaryGPU){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int icel;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  double r;
  double rxI, ryI;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }


  //SPREAD IN THE X DIRECTION
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);


  double fact =  2 * volumeParticleGPU * densfluidGPU /
    (densfluidGPU * 
     (2 * volumeParticleGPU*volumeGPU*densfluidGPU + massParticleGPU));
		  
  
  double SDeltap = fact * vxboundaryGPU[i] ;

  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = dlx4 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlx4 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlx4 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlx4 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlx4 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlx3 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlx3 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlx3 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlx3 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx3 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx2 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx2 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx2 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx2 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx2 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx1 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx1 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx1 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlx1 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlx1 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlx0 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlx0 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlx0 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlx0 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlx0 * dly0 * SDeltap;






  //SPREAD IN THE Y DIRECTION
  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  SDeltap = fact * vyboundaryGPU[i] ;

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = dlx4 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlx4 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlx4 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlx4 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlx4 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlx3 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlx3 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlx3 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlx3 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx3 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx2 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx2 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx2 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx2 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx2 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx1 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx1 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx1 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlx1 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlx1 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlx0 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlx0 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlx0 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlx0 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlx0 * dly0 * SDeltap;

}





























__global__ void kernelCorrectionVQuasiNeutrallyBuoyant4pt_2_2D(cufftDoubleComplex* vxZ,
							       cufftDoubleComplex* vyZ,
							       cufftDoubleComplex* vzZ,
							       const double* fxboundaryGPU,
							       const double* fyboundaryGPU,
							       const double* fzboundaryGPU){
   
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  int vecino, particle;
  double fx = 0.f;
  double fy = 0.f;
  double fz = 0.f;

  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  int vecino1 = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino1);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino1);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  int vecino2 = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino2);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino2);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];    
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  int vecino3 = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino3);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino3);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  int vecino4 = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino4);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino4);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  int vecinopxpy = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxpy);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  int vecinopxmy = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxmy);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  int vecinomxpy = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxpy);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  int vecinomxmy = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxmy);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymy
  vecino = tex1Dfetch(texvecino1GPU, vecino1);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxmymy
  vecino = tex1Dfetch(texvecinomxmyGPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle ];
    fy -= fyboundaryGPU[particle ];
  }
  //Particles in Cell vecinomxmxmy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+ (nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+ (nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmx
  vecino = tex1Dfetch(texvecino2GPU, vecino2);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpypy
  vecino = tex1Dfetch(texvecinomxpyGPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypy
  vecino = tex1Dfetch(texvecino4GPU, vecino4);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpypy
  vecino = tex1Dfetch(texvecinopxpyGPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpx
  vecino = tex1Dfetch(texvecino3GPU, vecino3);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmymy
  vecino = tex1Dfetch(texvecinopxmyGPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }


  
  vxZ[i].x = -fx ;
  vyZ[i].x = -fy ;
  vzZ[i].x = -fz ; 

  vxZ[i].y = 0;
  vyZ[i].y = 0;
  vzZ[i].y = 0;

}

























//First, spread S*(\Delta p - m_e*J*\Delta \tilde{ v })
__global__ void kernelSpreadDeltapMinusJTildev4pt_2D(const double* rxcellGPU,
						     const double* rycellGPU,
						     const double* rzcellGPU,
						     const cufftDoubleComplex* vxZ,
						     const cufftDoubleComplex* vyZ,
						     const cufftDoubleComplex* vzZ,
						     const double* vxboundaryGPU,
						     const double* vyboundaryGPU,
						     const double* vzboundaryGPU,
						     double* fxboundaryGPU,
						     double* fyboundaryGPU,
						     double* fzboundaryGPU){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  int icel;

  double r;
  double rxI, ryI;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }


  //SPREAD IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double tildev = dlx0 * dly0 * vxZ[vecinomxmxmymy].y +
    dlx0 * dly1 * vxZ[vecinomxmxmy].y +
    dlx0 * dly2 * vxZ[vecinomxmx].y +
    dlx0 * dly3 * vxZ[vecinomxmxpy].y +
    dlx0 * dly4 * vxZ[vecinomxmxpypy].y +
    dlx1 * dly0 * vxZ[vecinomxmymy].y +
    dlx1 * dly1 * vxZ[vecinomxmy].y +
    dlx1 * dly2 * vxZ[vecino2].y +
    dlx1 * dly3 * vxZ[vecinomxpy].y +
    dlx1 * dly4 * vxZ[vecinomxpypy].y +
    dlx2 * dly0 * vxZ[vecinomymy].y +
    dlx2 * dly1 * vxZ[vecino1].y +
    dlx2 * dly2 * vxZ[icel].y +
    dlx2 * dly3 * vxZ[vecino4].y +
    dlx2 * dly4 * vxZ[vecinopypy].y +
    dlx3 * dly0 * vxZ[vecinopxmymy].y +
    dlx3 * dly1 * vxZ[vecinopxmy].y +
    dlx3 * dly2 * vxZ[vecino3].y +
    dlx3 * dly3 * vxZ[vecinopxpy].y +
    dlx3 * dly4 * vxZ[vecinopxpypy].y;
    

  double SDeltap = (vxboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU);

  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = dlx4 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = dlx4 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlx4 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = dlx4 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = dlx4 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlx3 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = dlx3 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = dlx3 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlx3 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = dlx3 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = dlx2 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx2 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = dlx2 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = dlx2 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx2 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = dlx1 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = dlx1 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlx1 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = dlx1 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = dlx1 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlx0 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = dlx0 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = dlx0 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlx0 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = dlx0 * dly0 * SDeltap;



  //SPREAD IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  tildev = dlx0 * dly0 * vyZ[vecinomxmxmymy].y +
    dlx0 * dly1 * vyZ[vecinomxmxmy].y +
    dlx0 * dly2 * vyZ[vecinomxmx].y +
    dlx0 * dly3 * vyZ[vecinomxmxpy].y +
    dlx1 * dly0 * vyZ[vecinomxmymy].y +
    dlx1 * dly1 * vyZ[vecinomxmy].y +
    dlx1 * dly2 * vyZ[vecino2].y +
    dlx1 * dly3 * vyZ[vecinomxpy].y +
    dlx2 * dly0 * vyZ[vecinomymy].y +
    dlx2 * dly1 * vyZ[vecino1].y +
    dlx2 * dly2 * vyZ[icel].y +
    dlx2 * dly3 * vyZ[vecino4].y +
    dlx3 * dly0 * vyZ[vecinopxmymy].y +
    dlx3 * dly1 * vyZ[vecinopxmy].y +
    dlx3 * dly2 * vyZ[vecino3].y +
    dlx3 * dly3 * vyZ[vecinopxpy].y +
    dlx4 * dly0 * vyZ[vecinopxpxmymy].y +
    dlx4 * dly1 * vyZ[vecinopxpxmy].y +
    dlx4 * dly2 * vyZ[vecinopxpx].y +
    dlx4 * dly3 * vyZ[vecinopxpxpy].y;

  SDeltap = (vyboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU) ;
  
  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = dlx4 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = dlx4 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlx4 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = dlx4 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = dlx4 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlx3 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = dlx3 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = dlx3 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlx3 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = dlx3 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = dlx2 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx2 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = dlx2 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = dlx2 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx2 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = dlx1 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = dlx1 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlx1 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = dlx1 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = dlx1 * dly0 * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlx0 * dly4 * SDeltap;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = dlx0 * dly3 * SDeltap;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = dlx0 * dly2 * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlx0 * dly1 * SDeltap;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = dlx0 * dly0 * SDeltap;
}























//For the first time step
//Calculate positions at n+1/2
//and save in rxboundaryPrediction
__global__ void findNeighborParticlesQuasiNeutrallyBuoyant4ptTEST4_3_2D(const particlesincell* pc, 
									int* errorKernel,
									const double* rxcellGPU,
									const double* rycellGPU,
									const double* rzcellGPU,
									const double* rxboundaryGPU,  //q^{n}
									const double* ryboundaryGPU, 
									const double* rzboundaryGPU,
									double* rxboundaryPredictionGPU,//q^{n+1/2}
									double* ryboundaryPredictionGPU, 
									double* rzboundaryPredictionGPU,
									const double* vxGPU, //v^{n+1/2}
									const double* vyGPU, 
									const double* vzGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i); //rxboundaryPredictionGPU
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  
  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;

  int icel;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;
    
    icel = jx + jy*mxGPU;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * vxGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vxGPU[vecinomxmxmy] +
    dlx0 * dly2 * vxGPU[vecinomxmx] +
    dlx0 * dly3 * vxGPU[vecinomxmxpy] +
    dlx0 * dly4 * vxGPU[vecinomxmxpypy] +
    dlx1 * dly0 * vxGPU[vecinomxmymy] +
    dlx1 * dly1 * vxGPU[vecinomxmy] +
    dlx1 * dly2 * vxGPU[vecino2] +
    dlx1 * dly3 * vxGPU[vecinomxpy] +
    dlx1 * dly4 * vxGPU[vecinomxpypy] +
    dlx2 * dly0 * vxGPU[vecinomymy] +
    dlx2 * dly1 * vxGPU[vecino1] +
    dlx2 * dly2 * vxGPU[icel] +
    dlx2 * dly3 * vxGPU[vecino4] +
    dlx2 * dly4 * vxGPU[vecinopypy] +
    dlx3 * dly0 * vxGPU[vecinopxmymy] +
    dlx3 * dly1 * vxGPU[vecinopxmy] +
    dlx3 * dly2 * vxGPU[vecino3] +
    dlx3 * dly3 * vxGPU[vecinopxpy] +
    dlx3 * dly4 * vxGPU[vecinopxpypy];
  
  rxboundaryPredictionGPU[i] = rxboundaryGPU[i] + 0.5 * dtGPU * v;


  //VELOCITY IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * vyGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vyGPU[vecinomxmxmy] +
    dlx0 * dly2 * vyGPU[vecinomxmx] +
    dlx0 * dly3 * vyGPU[vecinomxmxpy] +
    dlx1 * dly0 * vyGPU[vecinomxmymy] +
    dlx1 * dly1 * vyGPU[vecinomxmy] +
    dlx1 * dly2 * vyGPU[vecino2] +
    dlx1 * dly3 * vyGPU[vecinomxpy] +
    dlx2 * dly0 * vyGPU[vecinomymy] +
    dlx2 * dly1 * vyGPU[vecino1] +
    dlx2 * dly2 * vyGPU[icel] +
    dlx2 * dly3 * vyGPU[vecino4] +
    dlx3 * dly0 * vyGPU[vecinopxmymy] +
    dlx3 * dly1 * vyGPU[vecinopxmy] +
    dlx3 * dly2 * vyGPU[vecino3] +
    dlx3 * dly3 * vyGPU[vecinopxpy] +
    dlx4 * dly0 * vyGPU[vecinopxpxmymy] +
    dlx4 * dly1 * vyGPU[vecinopxpxmy] +
    dlx4 * dly2 * vyGPU[vecinopxpx] +
    dlx4 * dly3 * vyGPU[vecinopxpxpy];
  
  
  ryboundaryPredictionGPU[i] = ryboundaryGPU[i] + 0.5 * dtGPU * v;
  
 
}






















//Calculate 0.5*dt*nu*J*L*\Delta v^{k=2} and store it
//in vxboundaryPredictionGPU
__global__ void interpolateLaplacianDeltaV4pt_2_2D(const double* rxcellGPU,
						   const double* rycellGPU,
						   const double* rzcellGPU,
						   const cufftDoubleComplex* vxZ,
						   const cufftDoubleComplex* vyZ,
						   const cufftDoubleComplex* vzZ,
						   const double* rxboundaryPredictionGPU,
						   const double* ryboundaryPredictionGPU,
						   const double* rzboundaryPredictionGPU,
						   double* vxboundaryPredictionGPU,
						   double* vyboundaryPredictionGPU,
						   double* vzboundaryPredictionGPU){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     



  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i); //rxboundaryPredictionGPU
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  
  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;

  int icel;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * vxZ[vecinomxmxmymy].y +
    dlx0 * dly1 * vxZ[vecinomxmxmy].y +
    dlx0 * dly2 * vxZ[vecinomxmx].y +
    dlx0 * dly3 * vxZ[vecinomxmxpy].y +
    dlx0 * dly4 * vxZ[vecinomxmxpypy].y +
    dlx1 * dly0 * vxZ[vecinomxmymy].y +
    dlx1 * dly1 * vxZ[vecinomxmy].y +
    dlx1 * dly2 * vxZ[vecino2].y +
    dlx1 * dly3 * vxZ[vecinomxpy].y +
    dlx1 * dly4 * vxZ[vecinomxpypy].y +
    dlx2 * dly0 * vxZ[vecinomymy].y +
    dlx2 * dly1 * vxZ[vecino1].y +
    dlx2 * dly2 * vxZ[icel].y +
    dlx2 * dly3 * vxZ[vecino4].y +
    dlx2 * dly4 * vxZ[vecinopypy].y +
    dlx3 * dly0 * vxZ[vecinopxmymy].y +
    dlx3 * dly1 * vxZ[vecinopxmy].y +
    dlx3 * dly2 * vxZ[vecino3].y +
    dlx3 * dly3 * vxZ[vecinopxpy].y +
    dlx3 * dly4 * vxZ[vecinopxpypy].y;

  vxboundaryPredictionGPU[i] = v;

  

  //VELOCITY IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);
  
  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * vyZ[vecinomxmxmymy].y +
    dlx0 * dly1 * vyZ[vecinomxmxmy].y +
    dlx0 * dly2 * vyZ[vecinomxmx].y +
    dlx0 * dly3 * vyZ[vecinomxmxpy].y +
    dlx1 * dly0 * vyZ[vecinomxmymy].y +
    dlx1 * dly1 * vyZ[vecinomxmy].y +
    dlx1 * dly2 * vyZ[vecino2].y +
    dlx1 * dly3 * vyZ[vecinomxpy].y +
    dlx2 * dly0 * vyZ[vecinomymy].y +
    dlx2 * dly1 * vyZ[vecino1].y +
    dlx2 * dly2 * vyZ[icel].y +
    dlx2 * dly3 * vyZ[vecino4].y +
    dlx3 * dly0 * vyZ[vecinopxmymy].y +
    dlx3 * dly1 * vyZ[vecinopxmy].y +
    dlx3 * dly2 * vyZ[vecino3].y +
    dlx3 * dly3 * vyZ[vecinopxpy].y +
    dlx4 * dly0 * vyZ[vecinopxpxmymy].y +
    dlx4 * dly1 * vyZ[vecinopxpxmy].y +
    dlx4 * dly2 * vyZ[vecinopxpx].y +
    dlx4 * dly3 * vyZ[vecinopxpxpy].y;
  
  vyboundaryPredictionGPU[i] = v;  
 
 
}

























__global__ void kernelConstructWQuasiNeutrallyBuoyant4ptTEST5_2_2D(const double *vxPredictionGPU, 
								   const double *vyPredictionGPU, 
								   const double *vzPredictionGPU, 
								   cufftDoubleComplex *WxZ, 
								   cufftDoubleComplex *WyZ, 
								   cufftDoubleComplex *WzZ, 
								   const double *d_rand,
								   const double *fxboundaryGPU,
								   const double *fyboundaryGPU,
								   const double *fzboundaryGPU,
								   const double* advXGPU,
								   const double* advYGPU,
								   const double* advZGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy;
  double vx, vy;
  double vx1, vx2, vx3, vx4;
  double vy1, vy2, vy3, vy4;
  int vecino1, vecino2, vecino3, vecino4; 
  
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);


  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;
  
  //Previous Velocity
  wx += vx ;//- pressurea1GPU * dtGPU;//* 0.267261241912424385 * dtGPU;
  wy += vy ;//- pressurea1GPU * 0.534522483824848769 * dtGPU;
  
  //Advection part
  double advX, advY;
  advX = WxZ[i].x;
  advY = WyZ[i].x;


  //NOISE part
  double dnoise_sXX, dnoise_sXY;
  double dnoise_sYY;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 2*ncellsGPU] ;
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/2.;
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 2*ncellsGPU] ;
  dnoise_sYY = d_rand[vecino4 + 2*ncellsGPU] - dnoise_tr/2.;
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4GPU * dnoise_sXY;
  wy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_tr = d_rand[i] + d_rand[i + 2*ncellsGPU] ;
  dnoise_sXX = d_rand[i] - dnoise_tr/2.;
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 2*ncellsGPU] - dnoise_tr/2.;
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4GPU * dnoise_sXY;

  

  int vecino, particle;
  double fx = 0.f;
  double fy = 0.f;


  //Particle contribution
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  np = tex1Dfetch(texCountParticlesInCellX,vecino1);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino1);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  np = tex1Dfetch(texCountParticlesInCellX,vecino2);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino2);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];    
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  np = tex1Dfetch(texCountParticlesInCellX,vecino3);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino3);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  np = tex1Dfetch(texCountParticlesInCellX,vecino4);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino4);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  int vecinopxpy = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxpy);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  int vecinopxmy = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxmy);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  int vecinomxpy = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxpy);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  int vecinomxmy = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxmy);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymy
  vecino = tex1Dfetch(texvecino1GPU, vecino1);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxmymy
  vecino = tex1Dfetch(texvecinomxmyGPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle ];
    fy -= fyboundaryGPU[particle ];
  }
  //Particles in Cell vecinomxmxmy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+ (nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+ (nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmx
  vecino = tex1Dfetch(texvecino2GPU, vecino2);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpypy
  vecino = tex1Dfetch(texvecinomxpyGPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypy
  vecino = tex1Dfetch(texvecino4GPU, vecino4);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpypy
  vecino = tex1Dfetch(texvecinopxpyGPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpx
  vecino = tex1Dfetch(texvecino3GPU, vecino3);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmymy
  vecino = tex1Dfetch(texvecinopxmyGPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }





  WxZ[i].x = wx + dtGPU * ( fx / (volumeGPU*densfluidGPU) ) - advX  ;
  WyZ[i].x = wy + dtGPU * ( fy / (volumeGPU*densfluidGPU) ) - advY  ;
  WzZ[i].x = 0;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;



}






















//FOR STEP 5, CALCULATE \Delta p
//Calculate \Delta p and saved in vxboundaryGPU
__global__ void kernelCalculateDeltap4pt_2D(const double* rxcellGPU,
					    const double* rycellGPU,
					    const double* rzcellGPU,
					    double* vxboundaryGPU,
					    double* vyboundaryGPU,
					    double* vzboundaryGPU,
					    const double* vxboundaryPredictionGPU,
					    const double* vyboundaryPredictionGPU,
					    const double* vzboundaryPredictionGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  //Particle position at time n+1/2
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  int icel;
  
  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }



  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * fetch_double(texVxGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVxGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVxGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVxGPU,vecinomxmxpy) +
    dlx0 * dly4 * fetch_double(texVxGPU,vecinomxmxpypy) +
    dlx1 * dly0 * fetch_double(texVxGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVxGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVxGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVxGPU,vecinomxpy) +
    dlx1 * dly4 * fetch_double(texVxGPU,vecinomxpypy) +
    dlx2 * dly0 * fetch_double(texVxGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVxGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVxGPU,icel) +
    dlx2 * dly3 * fetch_double(texVxGPU,vecino4) +
    dlx2 * dly4 * fetch_double(texVxGPU,vecinopypy) +
    dlx3 * dly0 * fetch_double(texVxGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVxGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVxGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVxGPU,vecinopxpy) +
    dlx3 * dly4 * fetch_double(texVxGPU,vecinopxpypy);

  
  vxboundaryGPU[i] = massParticleGPU * ( vxboundaryGPU[i] - v - vxboundaryPredictionGPU[i] );


  //Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * fetch_double(texVyGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVyGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVyGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVyGPU,vecinomxmxpy) +
    dlx1 * dly0 * fetch_double(texVyGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVyGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVyGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVyGPU,vecinomxpy) +
    dlx2 * dly0 * fetch_double(texVyGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVyGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVyGPU,icel) +
    dlx2 * dly3 * fetch_double(texVyGPU,vecino4) +
    dlx3 * dly0 * fetch_double(texVyGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVyGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVyGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVyGPU,vecinopxpy) +
    dlx4 * dly0 * fetch_double(texVyGPU,vecinopxpxmymy) +
    dlx4 * dly1 * fetch_double(texVyGPU,vecinopxpxmy) +
    dlx4 * dly2 * fetch_double(texVyGPU,vecinopxpx) +
    dlx4 * dly3 * fetch_double(texVyGPU,vecinopxpxpy);

  vyboundaryGPU[i] = massParticleGPU * ( vyboundaryGPU[i] - v - vyboundaryPredictionGPU[i] );


}























//Update particle velocity if me=0
__global__ void updateParticleVelocityme04pt_2D(const double* rxcellGPU,
						const double* rycellGPU,
						const double* rzcellGPU,
						double* vxboundaryGPU,
						double* vyboundaryGPU,
						double* vzboundaryGPU){
					  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  int icel;

  double r;
  double rxI, ryI;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }

  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * fetch_double(texVxGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVxGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVxGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVxGPU,vecinomxmxpy) +
    dlx0 * dly4 * fetch_double(texVxGPU,vecinomxmxpypy) +
    dlx1 * dly0 * fetch_double(texVxGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVxGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVxGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVxGPU,vecinomxpy) +
    dlx1 * dly4 * fetch_double(texVxGPU,vecinomxpypy) +
    dlx2 * dly0 * fetch_double(texVxGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVxGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVxGPU,icel) +
    dlx2 * dly3 * fetch_double(texVxGPU,vecino4) +
    dlx2 * dly4 * fetch_double(texVxGPU,vecinopypy) +
    dlx3 * dly0 * fetch_double(texVxGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVxGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVxGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVxGPU,vecinopxpy) +
    dlx3 * dly4 * fetch_double(texVxGPU,vecinopxpypy);
  

  vxboundaryGPU[i] = v;


  //SPREAD IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * fetch_double(texVyGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVyGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVyGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVyGPU,vecinomxmxpy) +
    dlx1 * dly0 * fetch_double(texVyGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVyGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVyGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVyGPU,vecinomxpy) +
    dlx2 * dly0 * fetch_double(texVyGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVyGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVyGPU,icel) +
    dlx2 * dly3 * fetch_double(texVyGPU,vecino4) +
    dlx3 * dly0 * fetch_double(texVyGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVyGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVyGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVyGPU,vecinopxpy) +
    dlx4 * dly0 * fetch_double(texVyGPU,vecinopxpxmymy) +
    dlx4 * dly1 * fetch_double(texVyGPU,vecinopxpxmy) +
    dlx4 * dly2 * fetch_double(texVyGPU,vecinopxpx) +
    dlx4 * dly3 * fetch_double(texVyGPU,vecinopxpxpy);

  vyboundaryGPU[i] = v;
  
}
























//Update particle velocity if me!=0
__global__ void updateParticleVelocityme4pt_2D(const double* rxcellGPU,
					       const double* rycellGPU,
					       const double* rzcellGPU,
					       const cufftDoubleComplex* vxZ,
					       const cufftDoubleComplex* vyZ,
					       const cufftDoubleComplex* vzZ,
					       double* vxboundaryGPU,
					       double* vyboundaryGPU,
					       double* vzboundaryGPU,
					       const double* vxboundaryPredictionGPU,
					       const double* vyboundaryPredictionGPU,
					       const double* vzboundaryPredictionGPU){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;
  int icel;

  double r;
  double rxI, ryI;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }


  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * fetch_double(texVxGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVxGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVxGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVxGPU,vecinomxmxpy) +
    dlx0 * dly4 * fetch_double(texVxGPU,vecinomxmxpypy) +
    dlx1 * dly0 * fetch_double(texVxGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVxGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVxGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVxGPU,vecinomxpy) +
    dlx1 * dly4 * fetch_double(texVxGPU,vecinomxpypy) +
    dlx2 * dly0 * fetch_double(texVxGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVxGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVxGPU,icel) +
    dlx2 * dly3 * fetch_double(texVxGPU,vecino4) +
    dlx2 * dly4 * fetch_double(texVxGPU,vecinopypy) +
    dlx3 * dly0 * fetch_double(texVxGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVxGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVxGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVxGPU,vecinopxpy) +
    dlx3 * dly4 * fetch_double(texVxGPU,vecinopxpypy);


  double tildev = dlx0 * dly0 * vxZ[vecinomxmxmymy].y +
    dlx0 * dly1 * vxZ[vecinomxmxmy].y +
    dlx0 * dly2 * vxZ[vecinomxmx].y +
    dlx0 * dly3 * vxZ[vecinomxmxpy].y +
    dlx0 * dly4 * vxZ[vecinomxmxpypy].y +
    dlx1 * dly0 * vxZ[vecinomxmymy].y +
    dlx1 * dly1 * vxZ[vecinomxmy].y +
    dlx1 * dly2 * vxZ[vecino2].y +
    dlx1 * dly3 * vxZ[vecinomxpy].y +
    dlx1 * dly4 * vxZ[vecinomxpypy].y +
    dlx2 * dly0 * vxZ[vecinomymy].y +
    dlx2 * dly1 * vxZ[vecino1].y +
    dlx2 * dly2 * vxZ[icel].y +
    dlx2 * dly3 * vxZ[vecino4].y +
    dlx2 * dly4 * vxZ[vecinopypy].y +
    dlx3 * dly0 * vxZ[vecinopxmymy].y +
    dlx3 * dly1 * vxZ[vecinopxmy].y +
    dlx3 * dly2 * vxZ[vecino3].y +
    dlx3 * dly3 * vxZ[vecinopxpy].y +
    dlx3 * dly4 * vxZ[vecinopxpypy].y;
    

  vxboundaryGPU[i] = v + tildev + vxboundaryPredictionGPU[i];


  //SPREAD IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * fetch_double(texVyGPU,vecinomxmxmymy) +
    dlx0 * dly1 * fetch_double(texVyGPU,vecinomxmxmy) +
    dlx0 * dly2 * fetch_double(texVyGPU,vecinomxmx) +
    dlx0 * dly3 * fetch_double(texVyGPU,vecinomxmxpy) +
    dlx1 * dly0 * fetch_double(texVyGPU,vecinomxmymy) +
    dlx1 * dly1 * fetch_double(texVyGPU,vecinomxmy) +
    dlx1 * dly2 * fetch_double(texVyGPU,vecino2) +
    dlx1 * dly3 * fetch_double(texVyGPU,vecinomxpy) +
    dlx2 * dly0 * fetch_double(texVyGPU,vecinomymy) +
    dlx2 * dly1 * fetch_double(texVyGPU,vecino1) +
    dlx2 * dly2 * fetch_double(texVyGPU,icel) +
    dlx2 * dly3 * fetch_double(texVyGPU,vecino4) +
    dlx3 * dly0 * fetch_double(texVyGPU,vecinopxmymy) +
    dlx3 * dly1 * fetch_double(texVyGPU,vecinopxmy) +
    dlx3 * dly2 * fetch_double(texVyGPU,vecino3) +
    dlx3 * dly3 * fetch_double(texVyGPU,vecinopxpy) +
    dlx4 * dly0 * fetch_double(texVyGPU,vecinopxpxmymy) +
    dlx4 * dly1 * fetch_double(texVyGPU,vecinopxpxmy) +
    dlx4 * dly2 * fetch_double(texVyGPU,vecinopxpx) +
    dlx4 * dly3 * fetch_double(texVyGPU,vecinopxpxpy);

  tildev = dlx0 * dly0 * vyZ[vecinomxmxmymy].y +
    dlx0 * dly1 * vyZ[vecinomxmxmy].y +
    dlx0 * dly2 * vyZ[vecinomxmx].y +
    dlx0 * dly3 * vyZ[vecinomxmxpy].y +
    dlx1 * dly0 * vyZ[vecinomxmymy].y +
    dlx1 * dly1 * vyZ[vecinomxmy].y +
    dlx1 * dly2 * vyZ[vecino2].y +
    dlx1 * dly3 * vyZ[vecinomxpy].y +
    dlx2 * dly0 * vyZ[vecinomymy].y +
    dlx2 * dly1 * vyZ[vecino1].y +
    dlx2 * dly2 * vyZ[icel].y +
    dlx2 * dly3 * vyZ[vecino4].y +
    dlx3 * dly0 * vyZ[vecinopxmymy].y +
    dlx3 * dly1 * vyZ[vecinopxmy].y +
    dlx3 * dly2 * vyZ[vecino3].y +
    dlx3 * dly3 * vyZ[vecinopxpy].y +
    dlx4 * dly0 * vyZ[vecinopxpxmymy].y +
    dlx4 * dly1 * vyZ[vecinopxpxmy].y +
    dlx4 * dly2 * vyZ[vecinopxpx].y +
    dlx4 * dly3 * vyZ[vecinopxpxpy].y;

  vyboundaryGPU[i] = v + tildev + vyboundaryPredictionGPU[i];
    
}
























__global__ void findNeighborParticlesQuasiNeutrallyBuoyant4ptTEST4_2_2D(particlesincell* pc, 
									int* errorKernel,
									const double* rxcellGPU,
									const double* rycellGPU,
									const double* rzcellGPU,
									double* rxboundaryGPU,  //q^{n}
									double* ryboundaryGPU, 
									double* rzboundaryGPU,
									const double* rxboundaryPredictionGPU,
									const double* ryboundaryPredictionGPU, 
									const double* rzboundaryPredictionGPU,
									const double* vxGPU, //v^{n+1/2}
									const double* vyGPU, 
									const double* vzGPU){
  //q^{n+1/2} = rxboundaryPredictionGPU

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i); //rxboundaryPredictionGPU
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  
  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;

  int icel;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel = jx + jy*mxGPU;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * vxGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vxGPU[vecinomxmxmy] +
    dlx0 * dly2 * vxGPU[vecinomxmx] +
    dlx0 * dly3 * vxGPU[vecinomxmxpy] +
    dlx0 * dly4 * vxGPU[vecinomxmxpypy] +
    dlx1 * dly0 * vxGPU[vecinomxmymy] +
    dlx1 * dly1 * vxGPU[vecinomxmy] +
    dlx1 * dly2 * vxGPU[vecino2] +
    dlx1 * dly3 * vxGPU[vecinomxpy] +
    dlx1 * dly4 * vxGPU[vecinomxpypy] +
    dlx2 * dly0 * vxGPU[vecinomymy] +
    dlx2 * dly1 * vxGPU[vecino1] +
    dlx2 * dly2 * vxGPU[icel] +
    dlx2 * dly3 * vxGPU[vecino4] +
    dlx2 * dly4 * vxGPU[vecinopypy] +
    dlx3 * dly0 * vxGPU[vecinopxmymy] +
    dlx3 * dly1 * vxGPU[vecinopxmy] +
    dlx3 * dly2 * vxGPU[vecino3] +
    dlx3 * dly3 * vxGPU[vecinopxpy] +
    dlx3 * dly4 * vxGPU[vecinopxpypy];
      
  rxboundaryGPU[i] = rxboundaryGPU[i] + dtGPU * v;

  //VELOCITY IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * vyGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vyGPU[vecinomxmxmy] +
    dlx0 * dly2 * vyGPU[vecinomxmx] +
    dlx0 * dly3 * vyGPU[vecinomxmxpy] +
    dlx1 * dly0 * vyGPU[vecinomxmymy] +
    dlx1 * dly1 * vyGPU[vecinomxmy] +
    dlx1 * dly2 * vyGPU[vecino2] +
    dlx1 * dly3 * vyGPU[vecinomxpy] +
    dlx2 * dly0 * vyGPU[vecinomymy] +
    dlx2 * dly1 * vyGPU[vecino1] +
    dlx2 * dly2 * vyGPU[icel] +
    dlx2 * dly3 * vyGPU[vecino4] +
    dlx3 * dly0 * vyGPU[vecinopxmymy] +
    dlx3 * dly1 * vyGPU[vecinopxmy] +
    dlx3 * dly2 * vyGPU[vecino3] +
    dlx3 * dly3 * vyGPU[vecinopxpy] +
    dlx4 * dly0 * vyGPU[vecinopxpxmymy] +
    dlx4 * dly1 * vyGPU[vecinopxpxmy] +
    dlx4 * dly2 * vyGPU[vecinopxpx] +
    dlx4 * dly3 * vyGPU[vecinopxpxpy];

  ryboundaryGPU[i] = ryboundaryGPU[i] + dtGPU * v;

}


























//Calculate 0.5*dt*nu*J*L*\Delta v^{k=2} and store it
//in vxboundaryPredictionGPU
__global__ void interpolateLaplacianDeltaV4pt_2D(const double* rxcellGPU,
						 const double* rycellGPU,
						 const double* rzcellGPU,
						 const double* vxGPU,
						 const double* vyGPU,
						 const double* vzGPU,
						 const double* rxboundaryPredictionGPU,
						 const double* ryboundaryPredictionGPU,
						 const double* rzboundaryPredictionGPU,
						 double* vxboundaryPredictionGPU,
						 double* vyboundaryPredictionGPU,
						 double* vzboundaryPredictionGPU){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;     



  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i); //rxboundaryPredictionGPU
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  
  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;

  double r;
  double rxI, ryI;
  double dlx0, dlx1, dlx2, dlx3, dlx4;
  double dly0, dly1, dly2, dly3, dly4;

  int icel;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    rxI = r;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    ryI = r;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    icel = jx + jy*mxGPU;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icel);
  vecino2 = tex1Dfetch(texvecino2GPU, icel);
  vecino3 = tex1Dfetch(texvecino3GPU, icel);
  vecino4 = tex1Dfetch(texvecino4GPU, icel);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icel);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icel);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icel);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icel);
  int vecinomymy = tex1Dfetch(texvecino1GPU, vecino1);
  int vecinomxmymy = tex1Dfetch(texvecino2GPU, vecinomymy);
  int vecinomxmxmymy = tex1Dfetch(texvecino2GPU, vecinomxmymy);
  int vecinomxmxmy = tex1Dfetch(texvecino4GPU, vecinomxmxmymy);
  int vecinomxmx = tex1Dfetch(texvecino4GPU, vecinomxmxmy);
  int vecinomxmxpy = tex1Dfetch(texvecino4GPU, vecinomxmx);
  int vecinomxmxpypy = tex1Dfetch(texvecino4GPU, vecinomxmxpy);
  int vecinomxpypy = tex1Dfetch(texvecino3GPU, vecinomxmxpypy);
  int vecinopypy = tex1Dfetch(texvecino3GPU, vecinomxpypy);
  int vecinopxpypy = tex1Dfetch(texvecino3GPU, vecinopypy);
  int vecinopxmymy = tex1Dfetch(texvecino3GPU, vecinomymy);
   
  //r = (rx - rxcellGPU[icelx] - dxGPU*0.5); //
  r = (rxI - rxcellGPU[icel] + dxGPU*0.5); //distance to cell vecino2
  delta4ptGPU(r,dlx0,dlx1,dlx2,dlx3);
  dlx4=0;

  r = (ryI - rycellGPU[icel]);
  delta4pt2GPU(r,dly0,dly1,dly2,dly3,dly4);

  double v = dlx0 * dly0 * vxGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vxGPU[vecinomxmxmy] +
    dlx0 * dly2 * vxGPU[vecinomxmx] +
    dlx0 * dly3 * vxGPU[vecinomxmxpy] +
    dlx0 * dly4 * vxGPU[vecinomxmxpypy] +
    dlx1 * dly0 * vxGPU[vecinomxmymy] +
    dlx1 * dly1 * vxGPU[vecinomxmy] +
    dlx1 * dly2 * vxGPU[vecino2] +
    dlx1 * dly3 * vxGPU[vecinomxpy] +
    dlx1 * dly4 * vxGPU[vecinomxpypy] +
    dlx2 * dly0 * vxGPU[vecinomymy] +
    dlx2 * dly1 * vxGPU[vecino1] +
    dlx2 * dly2 * vxGPU[icel] +
    dlx2 * dly3 * vxGPU[vecino4] +
    dlx2 * dly4 * vxGPU[vecinopypy] +
    dlx3 * dly0 * vxGPU[vecinopxmymy] +
    dlx3 * dly1 * vxGPU[vecinopxmy] +
    dlx3 * dly2 * vxGPU[vecino3] +
    dlx3 * dly3 * vxGPU[vecinopxpy] +
    dlx3 * dly4 * vxGPU[vecinopxpypy];

  vxboundaryPredictionGPU[i] = v;

  
  //VELOCITY IN THE Y DIRECTION
  int vecinopxpxpy = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpx = tex1Dfetch(texvecino1GPU, vecinopxpxpy);
  int vecinopxpxmy = tex1Dfetch(texvecino1GPU, vecinopxpx);
  int vecinopxpxmymy = tex1Dfetch(texvecino1GPU, vecinopxpxmy);

  r = (rxI - rxcellGPU[icel]); 
  delta4pt2GPU(r,dlx0,dlx1,dlx2,dlx3,dlx4);

  r = (ryI - rycellGPU[icel] + 0.5*dyGPU);
  delta4ptGPU(r,dly0,dly1,dly2,dly3);
  dly4=0;

  v = dlx0 * dly0 * vyGPU[vecinomxmxmymy] +
    dlx0 * dly1 * vyGPU[vecinomxmxmy] +
    dlx0 * dly2 * vyGPU[vecinomxmx] +
    dlx0 * dly3 * vyGPU[vecinomxmxpy] +
    dlx1 * dly0 * vyGPU[vecinomxmymy] +
    dlx1 * dly1 * vyGPU[vecinomxmy] +
    dlx1 * dly2 * vyGPU[vecino2] +
    dlx1 * dly3 * vyGPU[vecinomxpy] +
    dlx2 * dly0 * vyGPU[vecinomymy] +
    dlx2 * dly1 * vyGPU[vecino1] +
    dlx2 * dly2 * vyGPU[icel] +
    dlx2 * dly3 * vyGPU[vecino4] +
    dlx3 * dly0 * vyGPU[vecinopxmymy] +
    dlx3 * dly1 * vyGPU[vecinopxmy] +
    dlx3 * dly2 * vyGPU[vecino3] +
    dlx3 * dly3 * vyGPU[vecinopxpy] +
    dlx4 * dly0 * vyGPU[vecinopxpxmymy] +
    dlx4 * dly1 * vyGPU[vecinopxpxmy] +
    dlx4 * dly2 * vyGPU[vecinopxpx] +
    dlx4 * dly3 * vyGPU[vecinopxpxpy];
  
  vyboundaryPredictionGPU[i] = v;  

}





























__global__ void kernelConstructWQuasiNeutrallyBuoyant4ptTEST5_3_2D(const double *vxPredictionGPU, 
								   const double *vyPredictionGPU, 
								   const double *vzPredictionGPU, 
								   cufftDoubleComplex *WxZ, 
								   cufftDoubleComplex *WyZ, 
								   cufftDoubleComplex *WzZ, 
								   const double *d_rand,
								   const double *fxboundaryGPU,
								   const double *fyboundaryGPU,
								   const double *fzboundaryGPU,
								   double* advXGPU,
								   double* advYGPU,
								   double* advZGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy;
  double vx, vy;
  double vx1, vx2, vx3, vx4;
  double vy1, vy2, vy3, vy4;
  int vecino1, vecino2, vecino3, vecino4; 
  int vecinopxmy;
  int vecinomxpy;
  double vxmxpy, vypxmy;

  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);


  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;

  //Previous Velocity
  wx += vx ;//- pressurea1GPU * dtGPU;//* 0.267261241912424385 * dtGPU;
  wy += vy ;//- pressurea1GPU * 0.534522483824848769 * dtGPU;  wz += vz ;//- pressurea1GPU * 0.801783725737273154 * dtGPU;
  
  //Advection part
  double advX, advY;
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY  = 0.25 * dtGPU * advY;



  //NOISE part
  double dnoise_sXX, dnoise_sXY;
  double dnoise_sYY;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 2*ncellsGPU] ;
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/2.;
  wx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 2*ncellsGPU] ;
  dnoise_sYY = d_rand[vecino4 + 2*ncellsGPU] - dnoise_tr/2.;
  wy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4GPU * dnoise_sXY;
  wy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_tr = d_rand[i] + d_rand[i + 2*ncellsGPU] ;
  dnoise_sXX = d_rand[i] - dnoise_tr/2.;
  wx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 2*ncellsGPU] - dnoise_tr/2.;
  wy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4GPU * dnoise_sXY;


  int vecino, particle;
  double fx = 0.f;
  double fy = 0.f;


  //Particle contribution
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  np = tex1Dfetch(texCountParticlesInCellX,vecino1);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino1);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  np = tex1Dfetch(texCountParticlesInCellX,vecino2);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino2);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];    
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  np = tex1Dfetch(texCountParticlesInCellX,vecino3);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino3);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  np = tex1Dfetch(texCountParticlesInCellX,vecino4);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino4);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  int vecinopxpy = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxpy);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  np = tex1Dfetch(texCountParticlesInCellX,vecinopxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinopxmy);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxpy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxpy);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  int vecinomxmy = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecinomxmy);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecinomxmy);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymy
  vecino = tex1Dfetch(texvecino1GPU, vecino1);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxmymy
  vecino = tex1Dfetch(texvecinomxmyGPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle ];
    fy -= fyboundaryGPU[particle ];
  }
  //Particles in Cell vecinomxmxmy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+ (nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+ (nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmx
  vecino = tex1Dfetch(texvecino2GPU, vecino2);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpy
  vecino = tex1Dfetch(texvecino2GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmxpypy
  vecino = tex1Dfetch(texvecinomxpyGPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinomxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypy
  vecino = tex1Dfetch(texvecino4GPU, vecino4);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypy
  vecino = tex1Dfetch(texvecino4GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpypy
  vecino = tex1Dfetch(texvecinopxpyGPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxpy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxpy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpx
  vecino = tex1Dfetch(texvecino3GPU, vecino3);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmy
  vecino = tex1Dfetch(texvecino3GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpxmymy
  vecino = tex1Dfetch(texvecinopxmyGPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmymy
  vecino = tex1Dfetch(texvecino1GPU, vecinopxmy);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }



  

  WxZ[i].x = wx + dtGPU * (fx / (volumeGPU*densfluidGPU)) - 1.5*advX + 0.5*advXGPU[i];
  WyZ[i].x = wy + dtGPU * (fy / (volumeGPU*densfluidGPU)) - 1.5*advY + 0.5*advYGPU[i];
  WzZ[i].x = 0.;


  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

  //Save advection for the next time step
  advXGPU[i] = advX ;
  advYGPU[i] = advY ;
  advZGPU[i] = 0 ;

}

