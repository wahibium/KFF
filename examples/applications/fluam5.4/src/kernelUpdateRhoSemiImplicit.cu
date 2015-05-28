__global__ void kernelUpdateRhoSemiImplicit(cufftDoubleComplex *vxZ,
					    const prefactorsFourier *pF,
					    const double omega1){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega1 = 1.70710678118654752;
  //double omega1 = 0.292893218813452476;

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));

  //Construct denominator
  //double denominator = 1 - pow((omega1 * dtGPU),2) * pressurea1GPU * L ;
  double denominator = 1 - (omega1*dtGPU)*(omega1*dtGPU) * pressurea1GPU * L;

  vxZ[i].x = vxZ[i].x / denominator;
  vxZ[i].y = vxZ[i].y / denominator;
  
  
  
}








__global__ void kernelUpdateRhoSemiImplicit_2(cufftDoubleComplex *vxZ,
					      const prefactorsFourier *pF,
					      const double omega4){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega4 = 1.70710678118654752;
  //double omega4 = 0.292893218813452476;

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));

  //Construct denominator
  //double denominator = 1 - pow((omega4*dtGPU),2) * pressurea1GPU * L;
  double denominator = 1 - (omega4*dtGPU)*(omega4*dtGPU) * pressurea1GPU * L;

  vxZ[i].x = vxZ[i].x / denominator;
  vxZ[i].y = vxZ[i].y / denominator;

  
}

