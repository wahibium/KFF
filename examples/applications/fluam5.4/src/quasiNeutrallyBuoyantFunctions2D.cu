//First, spread \Delta p, prefactor * S*{\Delta p}
__global__ void kernelSpreadDeltap2D(const double* rxcellGPU,
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

  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy;

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;

  double r, rp, rm;

  {

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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }


  //SPREAD IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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


  double fact =  2 * volumeParticleGPU * densfluidGPU /
    (densfluidGPU * 
     (2 * volumeParticleGPU*volumeGPU*densfluidGPU + massParticleGPU));
		  
  
  double SDeltap = fact * vxboundaryGPU[i] ;


  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = dlxp * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlx  * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlxm * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dlym * SDeltap;







  //SPREAD IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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


  SDeltap = fact * vyboundaryGPU[i] ;


  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = dlxp * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlx  * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlxm * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dlym * SDeltap;




}





















__device__ void forceBondedParticleParticleGPU_2D(const int i,
						  double& fx, //Pass by reference
						  double& fy,
						  const double rx,
						  const double ry,
						  const bondedForcesVariables* bFV){	

  double x, y;
  double r, r0;
  double kSpring;
  int index;


  //Particle-Particle Force
  int nBonds = bFV->bondsParticleParticleGPU[i];
  int offset = bFV->bondsParticleParticleOffsetGPU[i];
  
  
  for(int j=0;j<nBonds;j++){

    index = bFV->bondsIndexParticleParticleGPU[offset + j];

    //Particle bonded coordinates
    x = fetch_double(texrxboundaryGPU,nboundaryGPU+index);
    y = fetch_double(texryboundaryGPU,nboundaryGPU+index);
    
    //Equilibrium distance 
    r0 = bFV->r0ParticleParticleGPU[offset+j];

    //Spring constant
    kSpring = bFV->kSpringParticleParticleGPU[offset+j];

    if(r0==0){
      fx += -kSpring * (rx - x);
      fy += -kSpring * (ry - y);
    }  
    else{     //If r0!=0 calculate particle particle distance
      r = sqrt( (x-rx)*(x-rx) + (y-ry)*(y-ry) );
      if(r>0){//If r=0 -> f=0
	fx += -kSpring * (1 - r0/r) * (rx - x);
	fy += -kSpring * (1 - r0/r) * (ry - y);
      }
    }
  }











  //Particle-FixedPoint Force
  nBonds = bFV->bondsParticleFixedPointGPU[i];
  offset = bFV->bondsParticleFixedPointOffsetGPU[i];
  
  
  for(int j=0;j<nBonds;j++){
    

    //Fixed point coordinates
    x = bFV->rxFixedPointGPU[offset+j];
    y = bFV->ryFixedPointGPU[offset+j];
    
    //Equilibrium distance 
    r0 = bFV->r0ParticleFixedPointGPU[offset+j];

    //Spring constant
    kSpring = bFV->kSpringParticleFixedPointGPU[offset+j];
    
    if(r0==0){
      fx += -kSpring * (rx - x);
      fy += -kSpring * (ry - y);
    }  
    else{     //If r0!=0 calculate particle particle distance
      r = sqrt( (x-rx)*(x-rx) + (y-ry)*(y-ry) );
      if(r>0){//If r=0 -> f=0
	fx += -kSpring * (1 - r0/r) * (rx - x);
	fy += -kSpring * (1 - r0/r) * (ry - y);
      }
    }
  }

    
  return ;
}










//STEP 2: CALCULATE FORCES AND SPREAD THEM TO THE FLUID S^{n+1/2} * F^{n+1/2}
//Fill "countparticlesincellX" lists
//and spread particle force F 
//Fill "countparticlesincellX" lists
//and spread particle force F 
__global__ void kernelSpreadParticlesForce2D(const double* rxcellGPU, 
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



  
  if(particlesWallGPU){
    //INCLUDE WALL REPULSION HERE
    //We use a repulsive Lennard-Jones
    double sigmaWall = 2*dyGPU;
    double cutoffWall = 1.12246204830937302 * sigmaWall; // 2^{1/6} * sigmaWall

    //IMPORTANT, for particleWall lyGPU stores ly+2*dy
    if(ry<(-0.5*lyGPU+cutoffWall+dyGPU)){//Left wall
      
      double distance = (0.5*lyGPU-dyGPU) + ry; //distance >= 0
      fy += 48 * temperatureGPU * (pow((sigmaWall/distance),13) - 0.5*pow((sigmaWall/distance),7));
      
    }
    else if(ry>(0.5*lyGPU-cutoffWall-dyGPU)){//Right wall
      
      double distance = (0.5*lyGPU-dyGPU) - ry; //distance >= 0
      fy -= 48 * temperatureGPU * (pow((sigmaWall/distance),13) - 0.5*pow((sigmaWall/distance),7));
      
    }
  }





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
  double r, rp, rm;

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

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
  }

  np = atomicAdd(&pc->countparticlesincellX[icelx],1);
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


  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;


  //FORCE IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);


  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx  = tex1D(texDelta, fabs(r));
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


    
  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = -dlxp * dlyp * fx;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = -dlxp * dly  * fx;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = -dlxp * dlym * fx;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = -dlx  * dlyp * fx;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = -dlx  * dly  * fx;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = -dlx  * dlym * fx;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = -dlxm * dlyp * fx;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = -dlxm * dly  * fx;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = -dlxm * dlym * fx;
  offset += nboundaryGPU+npGPU;//26

  
  
  //FORCE IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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



  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = -dlxp * dlyp * fy;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = -dlxp * dly  * fy;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = -dlxp * dlym * fy;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = -dlx  * dlyp * fy;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = -dlx  * dly  * fy;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = -dlx  * dlym * fy;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = -dlxm * dlyp * fy;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = -dlxm * dly  * fy;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = -dlxm * dlym * fy;



  

  
}



































__global__ void findNeighborParticlesQuasiNeutrallyBuoyant_1_2D(particlesincell* pc, 
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

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;

  int icelx, icely;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
 
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  
  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

    
  double v = dlxm * dlym * vxGPU[vecinomxmy] +
    dlxm * dly  * vxGPU[vecino2] +
    dlxm * dlyp * vxGPU[vecinomxpy] +
    dlx  * dlym * vxGPU[vecino1] +
    dlx  * dly  * vxGPU[icelx] + 
    dlx  * dlyp * vxGPU[vecino4] + 
    dlxp * dlym * vxGPU[vecinopxmy] + 
    dlxp * dly  * vxGPU[vecino3] +
    dlxp * dlyp * vxGPU[vecinopxpy] ;


  
  double rxNew = rx + 0.5 * dtGPU * v;
  rxboundaryPredictionGPU[nboundaryGPU+i] = rxNew;


  //VELOCITY IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  v = dlxm * dlym * vyGPU[vecinomxmy] +
    dlxm * dly  * vyGPU[vecino2] +
    dlxm * dlyp * vyGPU[vecinomxpy] +
    dlx  * dlym * vyGPU[vecino1] +
    dlx  * dly  * vyGPU[icely] + 
    dlx  * dlyp * vyGPU[vecino4] + 
    dlxp * dlym * vyGPU[vecinopxmy] + 
    dlxp * dly  * vyGPU[vecino3] +
    dlxp * dlyp * vyGPU[vecinopxpy] ;
  

  double ryNew = ry + 0.5 * dtGPU * v;

  ryboundaryPredictionGPU[nboundaryGPU+i] = ryNew;
 

  int icel;
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
























//FOR STEP 4
//Calculate \delta u^{n+1/2} and saved in vxboundaryPredictionGPU
__global__ void kernelCalculateDeltau_2D(const double* rxcellGPU,
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

  double r, rp, rm;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;


  //FIRST CALCULATE J^{n+1/2} * v^n
  {
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


    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }





  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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


  double v = dlxm * dlym * vxGPU[vecinomxmy] +
    dlxm * dly  * vxGPU[vecino2] +
    dlxm * dlyp * vxGPU[vecinomxpy] +
    dlx  * dlym * vxGPU[vecino1] +
    dlx  * dly  * vxGPU[icelx] + 
    dlx  * dlyp * vxGPU[vecino4] + 
    dlxp * dlym * vxGPU[vecinopxmy] + 
    dlxp * dly  * vxGPU[vecino3] +
    dlxp * dlyp * vxGPU[vecinopxpy];

  
  dux = v;


  //Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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


  v = dlxm * dlym * vyGPU[vecinomxmy] +
    dlxm * dly  * vyGPU[vecino2] +
    dlxm * dlyp * vyGPU[vecinomxpy] +
    dlx  * dlym * vyGPU[vecino1] +
    dlx  * dly  * vyGPU[icely] + 
    dlx  * dlyp * vyGPU[vecino4] + 
    dlxp * dlym * vyGPU[vecinopxmy] + 
    dlxp * dly  * vyGPU[vecino3] +
    dlxp * dlyp * vyGPU[vecinopxpy] ;    

  
  duy = v;




  //SECOND CALCULATE J^{n} * v^n

  //Particle positon at time n
  rx = rxboundaryGPU[i];
  ry = ryboundaryGPU[i];


  {
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


    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
  }





  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  //DEFINE MORE NEIGHBORS
  vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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

  v = dlxm * dlym * vxGPU[vecinomxmy] +
    dlxm * dly  * vxGPU[vecino2] +
    dlxm * dlyp * vxGPU[vecinomxpy] +
    dlx  * dlym * vxGPU[vecino1] +
    dlx  * dly  * vxGPU[icelx] + 
    dlx  * dlyp * vxGPU[vecino4] + 
    dlxp * dlym * vxGPU[vecinopxmy] + 
    dlxp * dly  * vxGPU[vecino3] +
    dlxp * dlyp * vxGPU[vecinopxpy] ;
    

  dux -= v;


  //Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);
  
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

  
  v = dlxm * dlym * vyGPU[vecinomxmy] +
    dlxm * dly  * vyGPU[vecino2] +
    dlxm * dlyp * vyGPU[vecinomxpy] +
    dlx  * dlym * vyGPU[vecino1] +
    dlx  * dly  * vyGPU[icely] + 
    dlx  * dlyp * vyGPU[vecino4] + 
    dlxp * dlym * vyGPU[vecinopxmy] + 
    dlxp * dly  * vyGPU[vecino3] +
    dlxp * dlyp * vyGPU[vecinopxpy] ;
    

  
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
__global__ void kernelCalculateDeltapFirstStep_2D(const double* rxcellGPU,
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

  double r, rp, rm;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;


  
  {

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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }





  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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


  double v = dlxm * dlym * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVxGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVxGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVxGPU,vecino1) +
    dlx  * dly  * fetch_double(texVxGPU,icelx) + 
    dlx  * dlyp * fetch_double(texVxGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVxGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVxGPU,vecinopxpy) ;
    

  
  vxboundaryPredictionGPU[i] = massParticleGPU * ( vxboundaryGPU[i] - v - vxboundaryPredictionGPU[i] );


  //Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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


  v = dlxm * dlym * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVyGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVyGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVyGPU,vecino1) +
    dlx  * dly  * fetch_double(texVyGPU,icely) + 
    dlx  * dlyp * fetch_double(texVyGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVyGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVyGPU,vecinopxpy) ;
    

  
  vyboundaryPredictionGPU[i] = massParticleGPU * ( vyboundaryGPU[i] - v - vyboundaryPredictionGPU[i] );


}



















//First, spread S*(\Delta p - m_e*J*\Delta \tilde{ v })
__global__ void kernelSpreadDeltapMinusJTildev_2D(const double* rxcellGPU,
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

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;

  double r, rp, rm;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }


  //SPREAD IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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

  double tildev = dlxm * dlym * vxZ[vecinomxmy].y +
    dlxm * dly  * vxZ[vecino2].y +
    dlxm * dlyp * vxZ[vecinomxpy].y +
    dlx  * dlym * vxZ[vecino1].y +
    dlx  * dly  * vxZ[icelx].y + 
    dlx  * dlyp * vxZ[vecino4].y + 
    dlxp * dlym * vxZ[vecinopxmy].y + 
    dlxp * dly  * vxZ[vecino3].y +
    dlxp * dlyp * vxZ[vecinopxpy].y ;
    

  double SDeltap = (vxboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU);


  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = dlxp * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = dlxp * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = dlxp * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = dlx  * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = dlx  * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = dlx  * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = dlxm * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = dlxm * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = dlxm * dlym * SDeltap;





  //SPREAD IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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

  tildev = dlxm * dlym * vyZ[vecinomxmy].y +
    dlxm * dly  * vyZ[vecino2].y +
    dlxm * dlyp * vyZ[vecinomxpy].y +
    dlx  * dlym * vyZ[vecino1].y +
    dlx  * dly  * vyZ[icely].y + 
    dlx  * dlyp * vyZ[vecino4].y + 
    dlxp * dlym * vyZ[vecinopxmy].y + 
    dlxp * dly  * vyZ[vecino3].y +
    dlxp * dlyp * vyZ[vecinopxpy].y ;
    


  SDeltap = (vyboundaryGPU[i] - massParticleGPU * tildev) / (densfluidGPU * volumeGPU) ;
  
  


  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = dlxp * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = dlxp * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = dlxp * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = dlx  * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = dlx  * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = dlx  * dlym * SDeltap;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = dlxm * dlyp * SDeltap;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = dlxm * dly  * SDeltap;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = dlxm * dlym * SDeltap;





}




















//For the first time step
//Calculate positions at n+1/2
//and save in rxboundaryPrediction
__global__ void findNeighborParticlesQuasiNeutrallyBuoyantTEST4_3_2D(const particlesincell* pc, 
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

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;

  int icelx, icely;

  {

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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);


  double v = dlxm * dlym * vxGPU[vecinomxmy] +
    dlxm * dly  * vxGPU[vecino2] +
    dlxm * dlyp * vxGPU[vecinomxpy] +
    dlx  * dlym * vxGPU[vecino1] +
    dlx  * dly  * vxGPU[icelx] + 
    dlx  * dlyp * vxGPU[vecino4] + 
    dlxp * dlym * vxGPU[vecinopxmy] + 
    dlxp * dly  * vxGPU[vecino3] +
    dlxp * dlyp * vxGPU[vecinopxpy] ;
    

  
  rxboundaryPredictionGPU[i] = rxboundaryGPU[i] + 0.5 * dtGPU * v;


  //VELOCITY IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);


  v = dlxm * dlym * vyGPU[vecinomxmy] +
    dlxm * dly  * vyGPU[vecino2] +
    dlxm * dlyp * vyGPU[vecinomxpy] +
    dlx  * dlym * vyGPU[vecino1] +
    dlx  * dly  * vyGPU[icely] + 
    dlx  * dlyp * vyGPU[vecino4] + 
    dlxp * dlym * vyGPU[vecinopxmy] + 
    dlxp * dly  * vyGPU[vecino3] +
    dlxp * dlyp * vyGPU[vecinopxpy] ;
    
  
  
  ryboundaryPredictionGPU[i] = ryboundaryGPU[i] + 0.5 * dtGPU * v;
  
 
}




























//Calculate 0.5*dt*nu*J*L*\Delta v^{k=2} and store it
//in vxboundaryPredictionGPU
__global__ void interpolateLaplacianDeltaV_2_2D(const double* rxcellGPU,
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

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;

  int icelx, icely;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  double v = dlxm * dlym * vxZ[vecinomxmy].y +
    dlxm * dly  * vxZ[vecino2].y +
    dlxm * dlyp * vxZ[vecinomxpy].y +
    dlx  * dlym * vxZ[vecino1].y +
    dlx  * dly  * vxZ[icelx].y + 
    dlx  * dlyp * vxZ[vecino4].y + 
    dlxp * dlym * vxZ[vecinopxmy].y + 
    dlxp * dly  * vxZ[vecino3].y +
    dlxp * dlyp * vxZ[vecinopxpy].y ;
    

  
  vxboundaryPredictionGPU[i] = v;

  

  //VELOCITY IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);


  v = dlxm * dlym * vyZ[vecinomxmy].y +
    dlxm * dly  * vyZ[vecino2].y +
    dlxm * dlyp * vyZ[vecinomxpy].y +
    dlx  * dlym * vyZ[vecino1].y +
    dlx  * dly  * vyZ[icely].y + 
    dlx  * dlyp * vyZ[vecino4].y + 
    dlxp * dlym * vyZ[vecinopxmy].y + 
    dlxp * dly  * vyZ[vecino3].y +
    dlxp * dlyp * vyZ[vecinopxpy].y ;

  
  vyboundaryPredictionGPU[i] = v;  
 
 
}

























//FOR STEP 5, CALCULATE \Delta p
//Calculate \Delta p and saved in vxboundaryGPU
__global__ void kernelCalculateDeltap_2D(const double* rxcellGPU,
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

  double r, rp, rm;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;
  
  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
  }



  //X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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

  double v = dlxm * dlym * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVxGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVxGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVxGPU,vecino1) +
    dlx  * dly  * fetch_double(texVxGPU,icelx) + 
    dlx  * dlyp * fetch_double(texVxGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVxGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVxGPU,vecinopxpy) ;
    

  
  vxboundaryGPU[i] = massParticleGPU * ( vxboundaryGPU[i] - v - vxboundaryPredictionGPU[i] );


  //Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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


  v = dlxm * dlym * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVyGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVyGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVyGPU,vecino1) +
    dlx  * dly  * fetch_double(texVyGPU,icely) + 
    dlx  * dlyp * fetch_double(texVyGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVyGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVyGPU,vecinopxpy) ;


  
  vyboundaryGPU[i] = massParticleGPU * ( vyboundaryGPU[i] - v - vyboundaryPredictionGPU[i] );


}
























//Update particle velocity if me=0
__global__ void updateParticleVelocityme0_2D(const double* rxcellGPU,
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

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;

  double r, rp, rm;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
  }


  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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


  double v = dlxm * dlym * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVxGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVxGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVxGPU,vecino1) +
    dlx  * dly  * fetch_double(texVxGPU,icelx) + 
    dlx  * dlyp * fetch_double(texVxGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVxGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVxGPU,vecinopxpy) ;
    


  vxboundaryGPU[i] = v;


  //SPREAD IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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


  v = dlxm * dlym * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVyGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVyGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVyGPU,vecino1) +
    dlx  * dly  * fetch_double(texVyGPU,icely) + 
    dlx  * dlyp * fetch_double(texVyGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVyGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVyGPU,vecinopxpy) ;
    


  vyboundaryGPU[i] = v;
  
}























//Update particle velocity if me!=0
__global__ void updateParticleVelocityme_2D(const double* rxcellGPU,
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

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  int icelx, icely;

  double r, rp, rm;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;

  }


  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);

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

 
  double v = dlxm * dlym * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVxGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVxGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVxGPU,vecino1) +
    dlx  * dly  * fetch_double(texVxGPU,icelx) + 
    dlx  * dlyp * fetch_double(texVxGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVxGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVxGPU,vecinopxpy) ;
    


  double tildev = dlxm * dlym * vxZ[vecinomxmy].y +
    dlxm * dly  * vxZ[vecino2].y +
    dlxm * dlyp * vxZ[vecinomxpy].y +
    dlx  * dlym * vxZ[vecino1].y +
    dlx  * dly  * vxZ[icelx].y + 
    dlx  * dlyp * vxZ[vecino4].y + 
    dlxp * dlym * vxZ[vecinopxmy].y + 
    dlxp * dly  * vxZ[vecino3].y +
    dlxp * dlyp * vxZ[vecinopxpy].y ;
    

  vxboundaryGPU[i] = v + tildev + vxboundaryPredictionGPU[i];


  //SPREAD IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

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

  v = dlxm * dlym * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dly  * fetch_double(texVyGPU,vecino2) +
    dlxm * dlyp * fetch_double(texVyGPU,vecinomxpy) +
    dlx  * dlym * fetch_double(texVyGPU,vecino1) +
    dlx  * dly  * fetch_double(texVyGPU,icely) + 
    dlx  * dlyp * fetch_double(texVyGPU,vecino4) + 
    dlxp * dlym * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dly  * fetch_double(texVyGPU,vecino3) +
    dlxp * dlyp * fetch_double(texVyGPU,vecinopxpy) ;
    


  tildev = dlxm * dlym * vyZ[vecinomxmy].y +
    dlxm * dly  * vyZ[vecino2].y +
    dlxm * dlyp * vyZ[vecinomxpy].y +
    dlx  * dlym * vyZ[vecino1].y +
    dlx  * dly  * vyZ[icely].y + 
    dlx  * dlyp * vyZ[vecino4].y + 
    dlxp * dlym * vyZ[vecinopxmy].y + 
    dlxp * dly  * vyZ[vecino3].y +
    dlxp * dlyp * vyZ[vecinopxpy].y ;
    


  vyboundaryGPU[i] = v + tildev + vyboundaryPredictionGPU[i];

    
}




























__global__ void findNeighborParticlesQuasiNeutrallyBuoyantTEST4_2_2D(particlesincell* pc, 
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

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;

  int icelx, icely;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  double v = dlxm * dlym * vxGPU[vecinomxmy] +
    dlxm * dly  * vxGPU[vecino2] +
    dlxm * dlyp * vxGPU[vecinomxpy] +
    dlx  * dlym * vxGPU[vecino1] +
    dlx  * dly  * vxGPU[icelx] + 
    dlx  * dlyp * vxGPU[vecino4] + 
    dlxp * dlym * vxGPU[vecinopxmy] + 
    dlxp * dly  * vxGPU[vecino3] +
    dlxp * dlyp * vxGPU[vecinopxpy] ;
    
  
  rxboundaryGPU[i] = rxboundaryGPU[i] + dtGPU * v;

  //VELOCITY IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);

  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);


  v = dlxm * dlym * vyGPU[vecinomxmy] +
    dlxm * dly  * vyGPU[vecino2] +
    dlxm * dlyp * vyGPU[vecinomxpy] +
    dlx  * dlym * vyGPU[vecino1] +
    dlx  * dly  * vyGPU[icely] + 
    dlx  * dlyp * vyGPU[vecino4] + 
    dlxp * dlym * vyGPU[vecinopxmy] + 
    dlxp * dly  * vyGPU[vecino3] +
    dlxp * dlyp * vyGPU[vecinopxpy] ;

  

  ryboundaryGPU[i] = ryboundaryGPU[i] + dtGPU * v;

}
























//Calculate 0.5*dt*nu*J*L*\Delta v^{k=2} and store it
//in vxboundaryPredictionGPU
__global__ void interpolateLaplacianDeltaV_2D(const double* rxcellGPU,
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

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;

  int icelx, icely;

  {
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

    icelx  = jxdx;
    icelx += jy * mxGPU;

    icely  = jx;
    icely += jydy * mxGPU;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  int vecinopxpx     = tex1Dfetch(texvecino3GPU, vecino3);
  int vecinopxpxpy   = tex1Dfetch(texvecino3GPU, vecinopxpy);
  int vecinopxpxmy   = tex1Dfetch(texvecino3GPU, vecinopxmy);
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  double v = dlxm * dlym * vxGPU[vecinomxmy] +
    dlxm * dly  * vxGPU[vecino2] +
    dlxm * dlyp * vxGPU[vecinomxpy] +
    dlx  * dlym * vxGPU[vecino1] +
    dlx  * dly  * vxGPU[icelx] + 
    dlx  * dlyp * vxGPU[vecino4] + 
    dlxp * dlym * vxGPU[vecinopxmy] + 
    dlxp * dly  * vxGPU[vecino3] +
    dlxp * dlyp * vxGPU[vecinopxpy] ;
    
  
  vxboundaryPredictionGPU[i] = v;

  

  //VELOCITY IN THE Y DIRECTION
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //DEFINE MORE NEIGHBORS
  int vecinopymxpy   = tex1Dfetch(texvecino4GPU, vecinomxpy);
  int vecinopypy     = tex1Dfetch(texvecino4GPU, vecino4);
  int vecinopypxpy   = tex1Dfetch(texvecino4GPU, vecinopxpy);
  
  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  
  v = dlxm * dlym * vyGPU[vecinomxmy] +
    dlxm * dly  * vyGPU[vecino2] +
    dlxm * dlyp * vyGPU[vecinomxpy] +
    dlx  * dlym * vyGPU[vecino1] +
    dlx  * dly  * vyGPU[icely] + 
    dlx  * dlyp * vyGPU[vecino4] + 
    dlxp * dlym * vyGPU[vecinopxmy] + 
    dlxp * dly  * vyGPU[vecino3] +
    dlxp * dlyp * vyGPU[vecinopxpy] ;
    
  
  vyboundaryPredictionGPU[i] = v;  
 
 

}


























//In this kernel we construct the vector W
//
// W = v^n + 0.5*dt*nu*L*v^n + Advection^{n+1/2} + (dt/rho)*f^n_{noise} + dt*SF/rho 
//
__global__ void kernelConstructWQuasiNeutrallyBuoyant_2D(const double *vxPredictionGPU, 
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
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
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
























__global__ void kernelCorrectionVQuasiNeutrallyBuoyant_2_2D(cufftDoubleComplex* vxZ,
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

  

  //Particles in Cell i
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
  }




  
  vxZ[i].x = -fx ;
  vyZ[i].x = -fy ;
  vzZ[i].x = -fz ; 

  vxZ[i].y = 0;
  vyZ[i].y = 0;
  vzZ[i].y = 0;

}





















__global__ void calculateAdvectionFluid_2D(const double* vxPredictionGPU,
					   const double* vyPredictionGPU,
					   const double* vzPredictionGPU,
					   cufftDoubleComplex* vxZ,
					   cufftDoubleComplex* vyZ,
					   cufftDoubleComplex* vzZ){
  

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  //for(int i=0;i<ncellsGPU;i++){
  
  int vecino1, vecino2, vecino3, vecino4; 
  int vecinopxmy;
  int vecinomxpy;

  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);

  double vx, vy;
  double vx1, vx2, vx3, vx4;
  double vy1, vy2, vy3, vy4;
  double vxmxpy;
  double vypxmy;
  
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vx1 = fetch_double(texVxGPU,vecino1);
  vy1 = fetch_double(texVyGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vy2 = fetch_double(texVyGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vy3 = fetch_double(texVyGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vy4 = fetch_double(texVyGPU,vecino4);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);

  //Advection part
  double advX, advY;
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY  = 0.25 * dtGPU * advY;


  vxZ[i].x = advX;
  vyZ[i].x = advY;
  vzZ[i].x = 0;



}



























__global__ void kernelConstructWQuasiNeutrallyBuoyantTEST5_2_2D(const double *vxPredictionGPU, 
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
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
  }  





  WxZ[i].x = wx + dtGPU * ( fx / (volumeGPU*densfluidGPU) ) - advX  ;
  WyZ[i].x = wy + dtGPU * ( fy / (volumeGPU*densfluidGPU) ) - advY  ;
  WzZ[i].x = 0;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;



}


















//Calculate nu*L*\Delta v^{k=2} and store it
//in vxGPU
__global__ void laplacianDeltaV_2D(const cufftDoubleComplex* vxZ,
				   const cufftDoubleComplex* vyZ,
				   const cufftDoubleComplex* vzZ,
				   double* vxGPU,
				   double* vyGPU,
				   double* vzGPU){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(ncellsGPU)) return;   

  int vecino1, vecino2, vecino3, vecino4; 
  
  double wx, wy;
  double vx, vy;
  double vx1, vx2, vx3, vx4;
  double vy1, vy2, vy3, vy4;

  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);

  vx = vxZ[i].x / double(ncellsGPU);
  vy = vyZ[i].x / double(ncellsGPU);
  vx1 = vxZ[vecino1].x / double(ncellsGPU);
  vx2 = vxZ[vecino2].x / double(ncellsGPU);
  vx3 = vxZ[vecino3].x / double(ncellsGPU);
  vx4 = vxZ[vecino4].x / double(ncellsGPU);
  vy1 = vyZ[vecino1].x / double(ncellsGPU);
  vy2 = vyZ[vecino2].x / double(ncellsGPU);
  vy3 = vyZ[vecino3].x / double(ncellsGPU);
  vy4 = vyZ[vecino4].x / double(ncellsGPU);

  //Laplacian part
  wx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  wx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  wx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wx;
  wy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  wy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  wy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * wy;

  vxGPU[i] = wx;
  vyGPU[i] = wy;
  vzGPU[i] = 0;

}
























__global__ void kernelConstructWQuasiNeutrallyBuoyantTEST5_3_2D(const double *vxPredictionGPU, 
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
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle];
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

