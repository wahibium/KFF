// Filename: runSchemeTestJPS.cu
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


bool runSchemeTestJPS(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;





  //Initialize textures cells
  if(!texturesCells()) return 0;  


  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,vecino2GPU,vecino3GPU,vecino4GPU,
						   vecinopxpyGPU,vecinopxmyGPU,vecinopxpzGPU,vecinopxmzGPU,
						   vecinomxpyGPU,vecinomxmyGPU,vecinomxpzGPU,vecinomxmzGPU,
						   vecinopypzGPU,vecinopymzGPU,vecinomypzGPU,vecinomymzGPU,
						   vecinopxpypzGPU,vecinopxpymzGPU,vecinopxmypzGPU,
						   vecinopxmymzGPU,
						   vecinomxpypzGPU,vecinomxpymzGPU,vecinomxmypzGPU,
						   vecinomxmymzGPU);
  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,vecino1GPU,vecino2GPU,
						    vecino3GPU,vecino4GPU,vecino5GPU);


  //Initialize plan
  cufftHandle FFT;
  cufftPlan3d(&FFT,mz,my,mx,CUFFT_Z2Z);

  //Initialize factors for fourier space update
  int threadsPerBlockdim, numBlocksdim;
  if((mx>=my)&&(mx>=mz)){
    threadsPerBlockdim = 128;
    numBlocksdim = (mx-1)/threadsPerBlockdim + 1;
  }
  else if((my>=mz)){
    threadsPerBlockdim = 128;
    numBlocksdim = (my-1)/threadsPerBlockdim + 1;
  }
  else{
    threadsPerBlockdim = 128;
    numBlocksdim = (mz-1)/threadsPerBlockdim + 1;
  }
  initializePrefactorFourierSpace_1<<<1,1>>>(gradKx,gradKy,gradKz,expKx,expKy,expKz,pF);
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);









  //JPS matrix
  double xx, xy, xz;
  double yx, yy, yz;
  double zx, zy, zz;
  
  //some JPS information
  double trace;
  double maxDiffDiag=0;
  double maxDiffNonDiag=0;
  double meanTrace=0;
  double errorTrace=0;
  //double meanMaxDiffDiag=0;
  //double meanMaxErrorNonDiag=0;
  int count=0;
  double traceXY, traceZ;
  double meanTraceXY=0;
  double meanTraceZ=0;
  double errorTraceXY=0;
  double errorTraceZ=0;

  //Point
  double x, y, z;

  //Initial point
  double rx0 = crx[0];
  double ry0 = cry[0];
  double rz0 = crz[0];
  
  //Distance between points
  double dx = lx/double(mx*testJPS*2);
  double dy = ly/double(my*testJPS*2);
  double dz = lz/double(mz*testJPS*2);
  
  //Loop to move the particle around
  // 
  for(int ix=0; ix<testJPS; ix++){
    for(int iy=0; iy<testJPS; iy++){
      for(int iz=0; iz<testJPS; iz++){

	x = rx0 + ix * dx;
	y = ry0 + iy * dy;
	z = rz0 + iz * dz;
	
	//Set fluid velocity field to zero
	setFieldToZeroInput<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU);

	//Spread vector (1,1,1)
	spreadVector<<<numBlocks,threadsPerBlock>>>(x,
						    y,
						    z,
						    1,
						    1,
						    1,
						    rxcellGPU,
						    rycellGPU,
						    rzcellGPU,
						    vxGPU,
						    vyGPU,
						    vzGPU);

	x = rx0 + ix * dx + pressurea0;
	y = ry0 + iy * dy + pressurea1;
	z = rz0 + iz * dz + pressurea2;

	
	//Apply incompressibility on S*(1,0,0)
	kernelConstructWTestJPS<<<numBlocks,threadsPerBlock>>>(0,//axis x
							       vxGPU,
							       vyGPU,
							       vzGPU,
							       vxZ,
							       vyZ,
							       vzZ);
	
	cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
	cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
	cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
	//Apply shift for the staggered grid
	kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
	//Apply incompressibility
	projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
	//Apply shift for the staggered grid
	kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
	cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
	cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
	cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

	//Interpolate
	interpolateField2<<<1,1>>>(rxcellGPU,
				   rycellGPU,
				   rzcellGPU,
				   vxZ,
				   vyZ,
				   vzZ,
				   x,
				   y,
				   z,
				   vxboundaryGPU,
				   vyboundaryGPU,
				   vzboundaryGPU);

	//Fill JPS matrix	
	cutilSafeCall(cudaMemcpy(vxParticle,&vxboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(vyParticle,&vyboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(vzParticle,&vzboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	
	xx=vxParticle[0];
	xy=vyParticle[0];
	xz=vzParticle[0];

	//Apply incompressibility on S*(0,1,0)
	kernelConstructWTestJPS<<<numBlocks,threadsPerBlock>>>(1,//axis x
							       vxGPU,
							       vyGPU,
							       vzGPU,
							       vxZ,
							       vyZ,
							       vzZ);
	
	cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
	cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
	cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
	//Apply shift for the staggered grid
	kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
	//Apply incompressibility
	projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
	//Apply shift for the staggered grid
	kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
	cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
	cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
	cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);


	//Interpolate
	interpolateField2<<<1,1>>>(rxcellGPU,
				   rycellGPU,
				   rzcellGPU,
				   vxZ,
				   vyZ,
				   vzZ,
				   x,
				   y,
				   z,
				   vxboundaryGPU,
				   vyboundaryGPU,
				   vzboundaryGPU);

	//Fill JPS matrix	
	cutilSafeCall(cudaMemcpy(vxParticle,&vxboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(vyParticle,&vyboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(vzParticle,&vzboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	
	yx=vxParticle[0];
	yy=vyParticle[0];
	yz=vzParticle[0];

	//Apply incompressibility on S*(0,0,1)
	kernelConstructWTestJPS<<<numBlocks,threadsPerBlock>>>(2,//axis x
							       vxGPU,
							       vyGPU,
							       vzGPU,
							       vxZ,
							       vyZ,
							       vzZ);
	
	cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
	cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
	cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
	//Apply shift for the staggered grid
	kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
	//Apply incompressibility
	projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
	//Apply shift for the staggered grid
	kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
	cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
	cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
	cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

	//Interpolate
	interpolateField2<<<1,1>>>(rxcellGPU,
				   rycellGPU,
				   rzcellGPU,
				   vxZ,
				   vyZ,
				   vzZ,
				   x,
				   y,
				   z,
				   vxboundaryGPU,
				   vyboundaryGPU,
				   vzboundaryGPU);

	//Fill JPS matrix	
	cutilSafeCall(cudaMemcpy(vxParticle,&vxboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(vyParticle,&vyboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(vzParticle,&vzboundaryGPU[nboundary],np*sizeof(double),cudaMemcpyDeviceToHost));
	
	zx=vxParticle[0];
	zy=vyParticle[0];
	zz=vzParticle[0];

	cout << ix << "  " << iy << "  " << iz << "  " << endl;
	cout << xx << "  " << xy << "  " << xz << "  " << endl;
	cout << yx << "  " << yy << "  " << yz << "  " << endl;
	cout << zx << "  " << zy << "  " << zz << "  " << endl;
	
	//Average JPS matrix
	trace = (xx + yy + zz) * 0.5; //We want the result to be 1
	errorTrace += count * (trace - meanTrace)*(trace - meanTrace) / (count+1);
	meanTrace += (trace-meanTrace) / (count+1);
	traceXY = (xx+yy) * 3 * 0.25; //We want the result to be 1
	errorTraceXY += count * (traceXY - meanTraceXY)*(traceXY - meanTraceXY) / (count+1);
	meanTraceXY += (traceXY-meanTraceXY) / (count+1);	
	traceZ = zz * 3 * 0.5; //We want the result to be 1
	errorTraceZ += count * (traceZ - meanTraceZ)*(traceZ - meanTraceZ) / (count+1);
	meanTraceZ += (traceZ-meanTraceZ) / (count+1);

	count++;

	//Modify
	if(abs(1.5*xx) > abs(maxDiffDiag)) maxDiffDiag = (1.5*xx);
	if(abs(1.5*yy) > abs(maxDiffDiag)) maxDiffDiag = (1.5*yy);
	if(abs(1.5*zz) > abs(maxDiffDiag)) maxDiffDiag = (1.5*zz);

	if(abs(xy) > abs(maxDiffNonDiag)) maxDiffNonDiag = (xy);
	if(abs(xz) > abs(maxDiffNonDiag)) maxDiffNonDiag = (xz);
	if(abs(yx) > abs(maxDiffNonDiag)) maxDiffNonDiag = (yx);
	if(abs(yz) > abs(maxDiffNonDiag)) maxDiffNonDiag = (yz);
	if(abs(zx) > abs(maxDiffNonDiag)) maxDiffNonDiag = (zx);
	if(abs(zy) > abs(maxDiffNonDiag)) maxDiffNonDiag = (zy);

	cout << "trace JPS                " << meanTrace << "   " << sqrt(errorTrace / count) << endl;
	cout << "max error diag, non-diag " << maxDiffDiag << "   " << 1.5*maxDiffNonDiag << endl << endl;
	cout << "trace JPS XY             " << meanTraceXY << "   " << sqrt(errorTraceXY / count) << endl;
	cout << "trace JPS Z              " << meanTraceZ << "   " << sqrt(errorTraceZ / count) << endl << endl;

	//Save some information of JPS matrix

      }
    }
  }
  


  //Free FFT
  cufftDestroy(FFT);
  cout << "trace JPS                " << meanTrace << "   " << sqrt(errorTrace / count) << endl;
  cout << "max error diag, non-diag " << maxDiffDiag << "   " << 1.5*maxDiffNonDiag << endl;
  cout << "trace JPS XY             " << meanTraceXY << "   " << sqrt(errorTraceXY / count) << endl;
  cout << "trace JPS Z              " << meanTraceZ << "   " << sqrt(errorTraceZ / count) << endl << endl;

  return 1;
}
