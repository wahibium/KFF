//STEP 1: g^* = g^n - omega1*dt*c^2*G*rho^* + A^n
//Calculate A^n
__global__ void kernelDpSemiImplicitCompressibleParticles_1(const double *densityGPU,
							    double *dpxGPU,
							    double *dpyGPU,
							    double *dpzGPU,
							    const double *d_rand,
							    const double *fxboundaryGPU,
							    const double *fyboundaryGPU,
							    const double *fzboundaryGPU,
							    const double omega1,
							    const double omega2,
							    const long long step){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega1 = 1.70710678118654752;
  //double omega1 = 0.292893218813452476;
  //double omega2 = 0.5;

  double fx = 0;
  double fy = 0;
  double fz = 0;

  double px = 0;
  double py = 0;
  double pz = 0;
  
  double sXX = 0;
  double sXY = 0;
  double sXZ = 0;
  double sYX = 0;
  double sYY = 0;
  double sYZ = 0;
  double sZX = 0;
  double sZY = 0;
  double sZZ = 0;


  //Particle contribution
  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int particle, vecino;
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+i);
    fz -= fzboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = tex1Dfetch(texvecino0GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
      fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
      fz -= fzboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }  
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino5
  vecino = tex1Dfetch(texvecino5GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = tex1Dfetch(texvecinopxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = tex1Dfetch(texvecinopxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = tex1Dfetch(texvecinomxpzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = tex1Dfetch(texvecinomxmzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = tex1Dfetch(texvecinopypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = tex1Dfetch(texvecinopymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = tex1Dfetch(texvecinomypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = tex1Dfetch(texvecinomymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = tex1Dfetch(texvecinopxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = tex1Dfetch(texvecinopxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = tex1Dfetch(texvecinopxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = tex1Dfetch(texvecinopxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = tex1Dfetch(texvecinomxpypzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = tex1Dfetch(texvecinomxpymzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = tex1Dfetch(texvecinomxmypzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = tex1Dfetch(texvecinomxmymzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle];
  }
  fx = -fx / (volumeGPU);
  fy = -fy / (volumeGPU);
  fz = -fz / (volumeGPU);



  double pressure, pressure3, pressure4, pressure5;
  double vx, vy, vz;

  double density = densityGPU[i];
  pressure = pressure_GPU(density);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 

  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopypz, vecinopymz, vecinomypz;
  int vecinopxpx, vecinopypy, vecinopzpz;

  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecinopxpx = tex1Dfetch(texvecino3GPU, vecino3);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecinopypy = tex1Dfetch(texvecino4GPU, vecino4);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopzpz = tex1Dfetch(texvecino5GPU, vecino5);

  vecinopxpy = tex1Dfetch(texvecinopxpyGPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopypz = tex1Dfetch(texvecinopypzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);



  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;

  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);

  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);

  double density0, density1, density2;
  double density3, density4, density5;

  density0 = densityGPU[vecino0];
  density1 = densityGPU[vecino1];
  density2 = densityGPU[vecino2];
  density3 = densityGPU[vecino3];
  pressure3 = pressure_GPU(density3);
  density4 = densityGPU[vecino4];
  pressure4 = pressure_GPU(density4);
  density5 = densityGPU[vecino5];
  pressure5 = pressure_GPU(density5);






  //Contribution from time n
  sXX += omega2 * fx * dxGPU ;//+ 0.00048828125 ;
  sYY += omega2 * fy * dyGPU ;
  sZZ += omega2 * fz * dzGPU ;


  pressure  = (omega2-omega1) * pressure;
  pressure3 = (omega2-omega1) * pressure3;
  pressure4 = (omega2-omega1) * pressure4;
  pressure5 = (omega2-omega1) * pressure5;

  //Include external pressure, explicit
  /*int kz = i / (mxmytGPU);
  double dp0 = 0.005;
  //double frequency=0.078413391828196923;//mz=32
  double frequency=0.0774325172412917947;//mz=32
  //Contribution at time n
  double time = step*dtGPU;
  if(kz==0){
    pressure  += omega2 * dp0 * sin(frequency*time);
    pressure3 += omega2 * dp0 * sin(frequency*time);
    pressure4 += omega2 * dp0 * sin(frequency*time);
  }
  else if(kz==(mzGPU-1)){
    pressure5 += omega2 * dp0 * sin(frequency*time);
    }*/

  sXX += (pressure3 - pressure);
  sYY += (pressure4 - pressure);
  sZZ += (pressure5 - pressure);





  sXX += omega2 * 0.125 * ((density3+density)*vx + (densityGPU[vecinopxpx]+density3)*vx3)*(vx+vx3);
  sXX -= omega2 * 0.125 * ((density+density2)*vx2 + (density3+density)*vx)*(vx2+vx);
  sYY += omega2 * 0.125 * ((density4+density)*vy + (densityGPU[vecinopypy]+density4)*vy4)*(vy+vy4);
  sYY -= omega2 * 0.125 * ((density+density1)*vy1 + (density4+density)*vy)*(vy1+vy);
  sZZ += omega2 * 0.125 * ((density5+density)*vz + (densityGPU[vecinopzpz]+density5)*vz5)*(vz+vz5);
  sZZ -= omega2 * 0.125 * ((density+density0)*vz0 + (density5+density)*vz)*(vz0+vz);

  double densitypxpy, densitypxmy, densitypxpz, densitypxmz;
  double densitymxpy, densitymxpz;
  double densitypypz, densitypymz, densitymypz;

  densitypxpy = densityGPU[vecinopxpy];
  densitypxmy = densityGPU[vecinopxmy];
  densitypxpz = densityGPU[vecinopxpz];
  densitypxmz = densityGPU[vecinopxmz];
  densitymxpy = densityGPU[vecinomxpy];
  densitymxpz = densityGPU[vecinomxpz];
  densitypypz = densityGPU[vecinopypz];
  densitypymz = densityGPU[vecinopymz];
  densitymypz = densityGPU[vecinomypz];

  
  sXY += omega2 * 0.125 * ((density4 + density)  * vy  + (densitypxpy + density3)    * vy3)    * (vx  + vx4);
  sXY -= omega2 * 0.125 * ((density  + density1) * vy1 + (density3    + densitypxmy) * vypxmy) * (vx1 + vx);
  sXZ += omega2 * 0.125 * ((density5 + density)  * vz  + (densitypxpz + density3)    * vz3)    * (vx  + vx5);
  sXZ -= omega2 * 0.125 * ((density  + density0) * vz0 + (density3    + densitypxmz) * vzpxmz) * (vx0 + vx);
  sYX += omega2 * 0.125 * ((density3 + density)  * vx  + (densitypxpy + density4)    * vx4)    * (vy  + vy3);
  sYX -= omega2 * 0.125 * ((density  + density2) * vx2 + (density4    + densitymxpy) * vxmxpy) * (vy2 + vy);
  sYZ += omega2 * 0.125 * ((density5 + density)  * vz  + (densitypypz + density4)    * vz4)    * (vy  + vy5);
  sYZ -= omega2 * 0.125 * ((density  + density0) * vz0 + (density4    + densitypymz) * vzpymz) * (vy0 + vy);
  sZX += omega2 * 0.125 * ((density3 + density)  * vx  + (densitypxpz + density5)    * vx5)    * (vz  + vz3);
  sZX -= omega2 * 0.125 * ((density  + density2) * vx2 + (density5    + densitymxpz) * vxmxpz) * (vz2 + vz);
  sZY += omega2 * 0.125 * ((density4 + density)  * vy  + (densitypypz + density5)    * vy5)    * (vz  + vz4);
  sZY -= omega2 * 0.125 * ((density  + density1) * vy1 + (density5    + densitymypz) * vymypz) * (vz1 + vz);


  //STRESS TENSOR CONTRIBUTION
  sXX -= omega2 * (2. * shearviscosityGPU * invdxGPU * (vx3 - vx) + fact3GPU *
		   (invdxGPU * (vx3 - vx) + invdyGPU * (vy3 - vypxmy) + invdzGPU * (vz3 - vzpxmz)));
  sXX += omega2 * (2. * shearviscosityGPU * invdxGPU * (vx - vx2) + fact3GPU *
		   (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));
  sYY -= omega2 * (2. * shearviscosityGPU * invdyGPU * (vy4 - vy) + fact3GPU * 
		   (invdxGPU * (vx4 - vxmxpy) + invdyGPU * (vy4 - vy) + invdzGPU * (vz4 - vzpymz)));
  sYY += omega2 * (2. * shearviscosityGPU * invdyGPU * (vy - vy1) + fact3GPU *
		   (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));
  sZZ -= omega2 * (2. * shearviscosityGPU * invdzGPU * (vz5 - vz) + fact3GPU *
		   (invdxGPU * (vx5 - vxmxpz) + invdyGPU * (vy5 - vymypz) + invdzGPU * (vz5 -vz)));
  sZZ += omega2 * (2. * shearviscosityGPU * invdzGPU * (vz - vz0) + fact3GPU *
		   (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));

  sXY -= omega2 * shearviscosityGPU * (invdyGPU * (vx4 - vx) + invdxGPU * (vy3 - vy));
  sXY += omega2 * shearviscosityGPU * (invdyGPU * (vx - vx1) + invdxGPU * (vypxmy - vy1));
  sXZ -= omega2 * shearviscosityGPU * (invdzGPU * (vx5 - vx) + invdxGPU * (vz3 - vz));
  sXZ += omega2 * shearviscosityGPU * (invdzGPU * (vx - vx0) + invdxGPU * (vzpxmz - vz0));
  sYX -= omega2 * shearviscosityGPU * (invdxGPU * (vy3 - vy) + invdyGPU * (vx4 - vx));
  sYX += omega2 * shearviscosityGPU * (invdxGPU * (vy - vy2) + invdyGPU * (vxmxpy - vx2));
  sYZ -= omega2 * shearviscosityGPU * (invdzGPU * (vy5 - vy) + invdyGPU * (vz4 - vz));
  sYZ += omega2 * shearviscosityGPU * (invdzGPU * (vy - vy0) + invdyGPU * (vzpymz - vz0));
  sZX -= omega2 * shearviscosityGPU * (invdxGPU * (vz3 - vz) + invdzGPU * (vx5 - vx));
  sZX += omega2 * shearviscosityGPU * (invdxGPU * (vz - vz2) + invdzGPU * (vxmxpz - vx2));
  sZY -= omega2 * shearviscosityGPU * (invdyGPU * (vz4 - vz) + invdzGPU * (vy5 - vy));
  sZY += omega2 * shearviscosityGPU * (invdyGPU * (vz - vz1) + invdzGPU * (vymypz - vy1));
  









  //Stress noise contribution
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  int n0;
  double fact1, fact2, fact4;

  fact1 = sqrt(omega2) * fact1GPU;
  fact2 = sqrt(omega2) * fact2GPU;
  fact4 = sqrt(omega2) * fact4GPU;

  n0 = ncellsGPU * 6;
  
  dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellsGPU] + d_rand[n0 + vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
  sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellsGPU] + d_rand[n0 + vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellsGPU] + d_rand[n0 + vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + i + ncellsGPU];
  sXY += fact4 * dnoise_sXY;
  sYX += fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + i + 2*ncellsGPU];
  sXZ += fact4 * dnoise_sXZ;
  sZX += fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + i + 4*ncellsGPU];
  sYZ += fact4 * dnoise_sYZ;
  sZY += fact4 * dnoise_sYZ;
  
  dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellsGPU] + d_rand[n0 + i + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
  sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_sYY = d_rand[n0 + i + 3*ncellsGPU] - dnoise_tr/3.;
  sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_sZZ = d_rand[n0 + i + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + vecino1 + ncellsGPU];
  sXY -= fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellsGPU];
  sXZ -= fact4 * dnoise_sXZ;
  
  dnoise_sXY = d_rand[n0 + vecino2 + ncellsGPU];
  sYX -= fact4 * dnoise_sXY;
  
  dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellsGPU];
  sYZ -= fact4 * dnoise_sYZ;
  
  dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellsGPU];
  sZX -= fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellsGPU];
  sZY -= fact4 * dnoise_sYZ;
  



  px = vx * 0.5 * (density + density3);
  py = vy * 0.5 * (density + density4);
  pz = vz * 0.5 * (density + density5);


  //px = fetch_double(texVxGPU,i) * 0.5 * (densityGPU[i] + densityGPU[vecino3]);
  //py = fetch_double(texVyGPU,i) * 0.5 * (densityGPU[i] + densityGPU[vecino4]);
  //pz = fetch_double(texVzGPU,i) * 0.5 * (densityGPU[i] + densityGPU[vecino5]);

  px += -(invdxGPU * sXX + invdyGPU * sXY + invdzGPU * sXZ)*dtGPU;
  py += -(invdxGPU * sYX + invdyGPU * sYY + invdzGPU * sYZ)*dtGPU;
  pz += -(invdxGPU * sZX + invdyGPU * sZY + invdzGPU * sZZ)*dtGPU;


  dpxGPU[i] = px;
  dpyGPU[i] = py;
  dpzGPU[i] = pz;




}



















//STEP 4: solve vx^*
__global__ void kernelDpSemiImplicitCompressibleParticles_2(double *vxPredictionGPU,
							    double *vyPredictionGPU,
							    double *vzPredictionGPU,
							    const double *dpxGPU,
							    const double *dpyGPU,
							    const double *dpzGPU,
							    const cufftDoubleComplex *vxZ,
							    const double omega1){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega1 = 1.70710678118654752;
  //double omega1 = 0.292893218813452476;
  

  int vecino3, vecino4, vecino5; 
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  
  double density  = vxZ[i].x / ncellsGPU;
  double density3 = vxZ[vecino3].x / ncellsGPU;
  double density4 = vxZ[vecino4].x / ncellsGPU;
  double density5 = vxZ[vecino5].x / ncellsGPU;

  double pressure  = pressure_GPU(density);
  double pressure3 = pressure_GPU(density3);
  double pressure4 = pressure_GPU(density4);
  double pressure5 = pressure_GPU(density5);
  
  

  //Explicit contribution
  double px = dpxGPU[i];
  double py = dpyGPU[i];
  double pz = dpzGPU[i];
  
  
  //Semi-implicit contribution
  double sXX = omega1 * (pressure3 - pressure) ;
  double sYY = omega1 * (pressure4 - pressure) ;
  double sZZ = omega1 * (pressure5 - pressure) ;


  px += -(invdxGPU * sXX)*dtGPU;
  py += -(invdyGPU * sYY)*dtGPU;
  pz += -(invdzGPU * sZZ)*dtGPU;


  vxPredictionGPU[i] = px * 2. / (density + density3);
  vyPredictionGPU[i] = py * 2. / (density + density4);
  vzPredictionGPU[i] = pz * 2. / (density + density5);

  //vxPredictionGPU[i] = fetch_double(texVxGPU,i);
  //vyPredictionGPU[i] = fetch_double(texVyGPU,i);
  //vzPredictionGPU[i] = fetch_double(texVzGPU,i);

  //vxZ[i].x = densityGPU[i] * ncellsGPU;
  //vxZ[i].y = 0;
  



}
















//STEP 5: g^{n+1} = g^n - omega1*dt*c^2*G*rho^{n+1} + A^{n+1/2}
//Calculate A^{n+1/2}
__global__ void kernelDpSemiImplicitCompressibleParticles_3(const double *densityGPU,
							    const double *vxPredictionGPU,
							    const double *vyPredictionGPU,
							    const double *vzPredictionGPU,
							    double *dpxGPU,
							    double *dpyGPU,
							    double *dpzGPU,
							    const double *d_rand,
							    const double *fxboundaryGPU,
							    const double *fyboundaryGPU,
							    const double *fzboundaryGPU,
							    const cufftDoubleComplex *vxZ,
							    const double omega2,
							    const double omega3,
							    const double omega4,
							    const double omega5,
							    const long long step){
  //const double *vxZ){
							          


  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega2 = 0.5;
  //double omega3 = -2.41421356237309505;
  //double omega3 = -0.414213562373095049;
  //double omega4 = 1.70710678118654752;
  //double omega4 = 0.292893218813452476;
  //double omega5 = 1;

  double fx = 0;
  double fy = 0;
  double fz = 0;

  double px = 0;
  double py = 0;
  double pz = 0;

  double sXX = 0;
  double sXY = 0;
  double sXZ = 0;
  double sYX = 0;
  double sYY = 0;
  double sYZ = 0;
  double sZX = 0;
  double sZY = 0;
  double sZZ = 0;

  //Particle contribution
  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int particle, vecino;
  int np = tex1Dfetch(texCountParticlesInCellX,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+i);
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+i);
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,i);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+i);
    fz -= fzboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = tex1Dfetch(texvecino0GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
      fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
      particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
      fz -= fzboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  //Particles in Cell vecino1
  vecino = tex1Dfetch(texvecino1GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = tex1Dfetch(texvecino2GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = tex1Dfetch(texvecino3GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }  
  //Particles in Cell vecino4
  vecino = tex1Dfetch(texvecino4GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino5
  vecino = tex1Dfetch(texvecino5GPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = tex1Dfetch(texvecinopxpyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = tex1Dfetch(texvecinopxmyGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = tex1Dfetch(texvecinopxpzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = tex1Dfetch(texvecinopxmzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = tex1Dfetch(texvecinomxpyGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = tex1Dfetch(texvecinomxmyGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = tex1Dfetch(texvecinomxpzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = tex1Dfetch(texvecinomxmzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = tex1Dfetch(texvecinopypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = tex1Dfetch(texvecinopymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = tex1Dfetch(texvecinomypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = tex1Dfetch(texvecinomymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = tex1Dfetch(texvecinopxpypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = tex1Dfetch(texvecinopxpymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = tex1Dfetch(texvecinopxmypzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = tex1Dfetch(texvecinopxmymzGPU, i);
  np = tex1Dfetch(texCountParticlesInCellX,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellX,ncellsGPU*j+vecino);
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellY,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellY,ncellsGPU*j+vecino);
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = tex1Dfetch(texvecinomxpypzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = tex1Dfetch(texvecinomxpymzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = tex1Dfetch(texvecinomxmypzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = tex1Dfetch(texvecinomxmymzGPU, i);
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
  np = tex1Dfetch(texCountParticlesInCellZ,vecino);
  for(int j=0;j<np;j++){
    particle = tex1Dfetch(texPartInCellZ,ncellsGPU*j+vecino);
    fz -= fzboundaryGPU[particle];
  }
  fx = -fx / (volumeGPU);
  fy = -fy / (volumeGPU);
  fz = -fz / (volumeGPU);


  double pressure, pressure3, pressure4, pressure5;
  double vx, vy, vz;

  double density = densityGPU[i];
  pressure = pressure_GPU(density);

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 

  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopypz, vecinopymz, vecinomypz;
  int vecinopxpx, vecinopypy, vecinopzpz;

  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecinopxpx = tex1Dfetch(texvecino3GPU, vecino3);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecinopypy = tex1Dfetch(texvecino4GPU, vecino4);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopzpz = tex1Dfetch(texvecino5GPU, vecino5);

  


  /*int kx, ky, kz;
  kz = vecino5 / (mxGPU*myGPU);
  ky = (vecino5 % (mxGPU*myGPU)) / mxGPU;
  kx = vecino5 % mxGPU;
  kz=(kz+1)%mzGPU;
  vecinopzpz = kx + ky*mxGPU + kz*mxGPU*myGPU;*/

  vecinopxpy = tex1Dfetch(texvecinopxpyGPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopypz = tex1Dfetch(texvecinopypzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);



  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;

  double vxmxpy, vxmxpz;
  double vypxmy, vymypz;
  double vzpxmz, vzpymz;

  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);

  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);

  double density0, density1, density2;
  double density3, density4, density5;

  density0 = densityGPU[vecino0];
  density1 = densityGPU[vecino1];
  density2 = densityGPU[vecino2];
  density3 = densityGPU[vecino3];
  pressure3 = pressure_GPU(density3);
  density4 = densityGPU[vecino4];
  pressure4 = pressure_GPU(density4);
  density5 = densityGPU[vecino5];
  pressure5 = pressure_GPU(density5);


  px = vx * 0.5 * (density + density3);
  py = vy * 0.5 * (density + density4);
  pz = vz * 0.5 * (density + density5);



  //Contribution from time n
  sXX += (1-omega5) * fx * dxGPU ;//+ 0.00048828125 ;
  sYY += (1-omega5) * fy * dyGPU ;
  sZZ += (1-omega5) * fz * dzGPU ;

  //sXX += (1-omega3-omega4) * (pressure3 - pressure);
  //sYY += (1-omega3-omega4) * (pressure4 - pressure);
  //sZZ += (1-omega3-omega4) * (pressure5 - pressure);
  pressure  = (1-omega3-omega4) * pressure;
  pressure3 = (1-omega3-omega4) * pressure3;
  pressure4 = (1-omega3-omega4) * pressure4;
  pressure5 = (1-omega3-omega4) * pressure5;



  //Include external pressure, explicit form
  /*int kz = i / (mxmytGPU);
  double dp0 = 0.005;
  //double frequency=0.078413391828196923;//mz=32
  double frequency=0.0774325172412917947;//mz=32
  //Contribution at time n
  double time = step*dtGPU;
  if(kz==0){
    pressure  += (1-omega5) * dp0 * sin(frequency*time);
    pressure3 += (1-omega5) * dp0 * sin(frequency*time);
    pressure4 += (1-omega5) * dp0 * sin(frequency*time);
  }
  else if(kz==(mzGPU-1)){
    pressure5 += (1-omega5) * dp0 * sin(frequency*time);
  }

  //Contribution at time n+omega2
  time = (step+omega2)*dtGPU;
  if(kz==0){
    pressure  += omega5 * dp0 * sin(frequency*time);
    pressure3 += omega5 * dp0 * sin(frequency*time);
    pressure4 += omega5 * dp0 * sin(frequency*time);
  }
  else if(kz==(mzGPU-1)){
    pressure5 += omega5 * dp0 * sin(frequency*time);
    }*/

  sXX += (pressure3 - pressure);
  sYY += (pressure4 - pressure);
  sZZ += (pressure5 - pressure);











  //NOTE: IN THIS IMPLEMENTATION OMEGA5=1, WE COMMENT THE NEXT LINES
  double densitypxpy, densitypxmy, densitypxpz, densitypxmz;
  double densitymxpy, densitymxpz;
  double densitypypz, densitypymz, densitymypz;
  
  sXX += (1-omega5) * 0.125 * ((density3+density)*vx + (densityGPU[vecinopxpx]+density3)*vx3)*(vx+vx3);
  sXX -= (1-omega5) * 0.125 * ((density+density2)*vx2 + (density3+density)*vx)*(vx2+vx);
  sYY += (1-omega5) * 0.125 * ((density4+density)*vy + (densityGPU[vecinopypy]+density4)*vy4)*(vy+vy4);
  sYY -= (1-omega5) * 0.125 * ((density+density1)*vy1 + (density4+density)*vy)*(vy1+vy);
  sZZ += (1-omega5) * 0.125 * ((density5+density)*vz + (densityGPU[vecinopzpz]+density5)*vz5)*(vz+vz5);
  sZZ -= (1-omega5) * 0.125 * ((density+density0)*vz0 + (density5+density)*vz)*(vz0+vz);


  densitypxpy = densityGPU[vecinopxpy];
  densitypxmy = densityGPU[vecinopxmy];
  densitypxpz = densityGPU[vecinopxpz];
  densitypxmz = densityGPU[vecinopxmz];
  densitymxpy = densityGPU[vecinomxpy];
  densitymxpz = densityGPU[vecinomxpz];
  densitypypz = densityGPU[vecinopypz];
  densitypymz = densityGPU[vecinopymz];
  densitymypz = densityGPU[vecinomypz];

  
  sXY += (1-omega5) * 0.125 * ((density4 + density)  * vy  + (densitypxpy + density3)    * vy3)    * (vx  + vx4);
  sXY -= (1-omega5) * 0.125 * ((density  + density1) * vy1 + (density3    + densitypxmy) * vypxmy) * (vx1 + vx);
  sXZ += (1-omega5) * 0.125 * ((density5 + density)  * vz  + (densitypxpz + density3)    * vz3)    * (vx  + vx5);
  sXZ -= (1-omega5) * 0.125 * ((density  + density0) * vz0 + (density3    + densitypxmz) * vzpxmz) * (vx0 + vx);
  sYX += (1-omega5) * 0.125 * ((density3 + density)  * vx  + (densitypxpy + density4)    * vx4)    * (vy  + vy3);
  sYX -= (1-omega5) * 0.125 * ((density  + density2) * vx2 + (density4    + densitymxpy) * vxmxpy) * (vy2 + vy);
  sYZ += (1-omega5) * 0.125 * ((density5 + density)  * vz  + (densitypypz + density4)    * vz4)    * (vy  + vy5);
  sYZ -= (1-omega5) * 0.125 * ((density  + density0) * vz0 + (density4    + densitypymz) * vzpymz) * (vy0 + vy);
  sZX += (1-omega5) * 0.125 * ((density3 + density)  * vx  + (densitypxpz + density5)    * vx5)    * (vz  + vz3);
  sZX -= (1-omega5) * 0.125 * ((density  + density2) * vx2 + (density5    + densitymxpz) * vxmxpz) * (vz2 + vz);
  sZY += (1-omega5) * 0.125 * ((density4 + density)  * vy  + (densitypypz + density5)    * vy5)    * (vz  + vz4);
  sZY -= (1-omega5) * 0.125 * ((density  + density1) * vy1 + (density5    + densitymypz) * vymypz) * (vz1 + vz);


  //STRESS TENSOR CONTRIBUTION
  sXX -= (1-omega5) * (2. * shearviscosityGPU * invdxGPU * (vx3 - vx) + fact3GPU *
		       (invdxGPU * (vx3 - vx) + invdyGPU * (vy3 - vypxmy) + invdzGPU * (vz3 - vzpxmz)));
  sXX += (1-omega5) * (2. * shearviscosityGPU * invdxGPU * (vx - vx2) + fact3GPU *
		       (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));
  sYY -= (1-omega5) * (2. * shearviscosityGPU * invdyGPU * (vy4 - vy) + fact3GPU * 
		       (invdxGPU * (vx4 - vxmxpy) + invdyGPU * (vy4 - vy) + invdzGPU * (vz4 - vzpymz)));
  sYY += (1-omega5) * (2. * shearviscosityGPU * invdyGPU * (vy - vy1) + fact3GPU *
		       (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));
  sZZ -= (1-omega5) * (2. * shearviscosityGPU * invdzGPU * (vz5 - vz) + fact3GPU *
		       (invdxGPU * (vx5 - vxmxpz) + invdyGPU * (vy5 - vymypz) + invdzGPU * (vz5 -vz)));
  sZZ += (1-omega5) * (2. * shearviscosityGPU * invdzGPU * (vz - vz0) + fact3GPU *
		       (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));

  sXY -= (1-omega5) * shearviscosityGPU * (invdyGPU * (vx4 - vx) + invdxGPU * (vy3 - vy));
  sXY += (1-omega5) * shearviscosityGPU * (invdyGPU * (vx - vx1) + invdxGPU * (vypxmy - vy1));
  sXZ -= (1-omega5) * shearviscosityGPU * (invdzGPU * (vx5 - vx) + invdxGPU * (vz3 - vz));
  sXZ += (1-omega5) * shearviscosityGPU * (invdzGPU * (vx - vx0) + invdxGPU * (vzpxmz - vz0));
  sYX -= (1-omega5) * shearviscosityGPU * (invdxGPU * (vy3 - vy) + invdyGPU * (vx4 - vx));
  sYX += (1-omega5) * shearviscosityGPU * (invdxGPU * (vy - vy2) + invdyGPU * (vxmxpy - vx2));
  sYZ -= (1-omega5) * shearviscosityGPU * (invdzGPU * (vy5 - vy) + invdyGPU * (vz4 - vz));
  sYZ += (1-omega5) * shearviscosityGPU * (invdzGPU * (vy - vy0) + invdyGPU * (vzpymz - vz0));
  sZX -= (1-omega5) * shearviscosityGPU * (invdxGPU * (vz3 - vz) + invdzGPU * (vx5 - vx));
  sZX += (1-omega5) * shearviscosityGPU * (invdxGPU * (vz - vz2) + invdzGPU * (vxmxpz - vx2));
  sZY -= (1-omega5) * shearviscosityGPU * (invdyGPU * (vz4 - vz) + invdzGPU * (vy5 - vy));
  sZY += (1-omega5) * shearviscosityGPU * (invdyGPU * (vz - vz1) + invdzGPU * (vymypz - vy1));
  






  //Contribution from time n+omega2
  density = vxZ[i].x / ncellsGPU;
  pressure = pressure_GPU(density);


  vx = vxPredictionGPU[i];
  vy = vyPredictionGPU[i];
  vz = vzPredictionGPU[i];
  vx0 = vxPredictionGPU[vecino0];
  vx1 = vxPredictionGPU[vecino1];
  vx2 = vxPredictionGPU[vecino2];
  vx3 = vxPredictionGPU[vecino3];
  vx4 = vxPredictionGPU[vecino4];
  vx5 = vxPredictionGPU[vecino5];
  vy0 = vyPredictionGPU[vecino0];
  vy1 = vyPredictionGPU[vecino1];
  vy2 = vyPredictionGPU[vecino2];
  vy3 = vyPredictionGPU[vecino3];
  vy4 = vyPredictionGPU[vecino4];
  vy5 = vyPredictionGPU[vecino5];
  vz0 = vzPredictionGPU[vecino0];
  vz1 = vzPredictionGPU[vecino1];
  vz2 = vzPredictionGPU[vecino2];
  vz3 = vzPredictionGPU[vecino3];
  vz4 = vzPredictionGPU[vecino4];
  vz5 = vzPredictionGPU[vecino5];

  vxmxpy = vxPredictionGPU[vecinomxpy];
  vxmxpz = vxPredictionGPU[vecinomxpz];
  vypxmy = vyPredictionGPU[vecinopxmy];
  vymypz = vyPredictionGPU[vecinomypz];
  vzpxmz = vzPredictionGPU[vecinopxmz];
  vzpymz = vzPredictionGPU[vecinopymz];

  density0 = vxZ[vecino0].x / ncellsGPU;
  density1 = vxZ[vecino1].x / ncellsGPU;
  density2 = vxZ[vecino2].x / ncellsGPU;
  density3 = vxZ[vecino3].x / ncellsGPU;
  pressure3 = pressure_GPU(density3);
  density4 = vxZ[vecino4].x / ncellsGPU;
  pressure4 = pressure_GPU(density4);
  density5 = vxZ[vecino5].x / ncellsGPU;
  pressure5 = pressure_GPU(density5);



  sXX += omega5 * fx * dxGPU ;//+ 0.00048828125 ;
  sYY += omega5 * fy * dyGPU ;
  sZZ += omega5 * fz * dzGPU ;



  sXX += omega3 * (pressure3 - pressure);
  sYY += omega3 * (pressure4 - pressure);
  sZZ += omega3 * (pressure5 - pressure);

  double densitypxpx = vxZ[vecinopxpx].x / ncellsGPU;
  double densitypypy = vxZ[vecinopypy].x / ncellsGPU;
  double densitypzpz = vxZ[vecinopzpz].x / ncellsGPU;

  sXX += omega5 * 0.125 * ((density3+density)*vx + (densitypxpx+density3)*vx3)*(vx+vx3);
  sXX -= omega5 * 0.125 * ((density+density2)*vx2 + (density3+density)*vx)*(vx2+vx);
  sYY += omega5 * 0.125 * ((density4+density)*vy + (densitypypy+density4)*vy4)*(vy+vy4);
  sYY -= omega5 * 0.125 * ((density+density1)*vy1 + (density4+density)*vy)*(vy1+vy);
  sZZ += omega5 * 0.125 * ((density5+density)*vz + (densitypzpz+density5)*vz5)*(vz+vz5);
  sZZ -= omega5 * 0.125 * ((density+density0)*vz0 + (density5+density)*vz)*(vz0+vz);

  /*double densitypxpy, densitypxmy, densitypxpz, densitypxmz;
  double densitymxpy, densitymxpz;
  double densitypypz, densitypymz, densitymypz;*/

  densitypxpy = vxZ[vecinopxpy].x / ncellsGPU;
  densitypxmy = vxZ[vecinopxmy].x / ncellsGPU;
  densitypxpz = vxZ[vecinopxpz].x / ncellsGPU;
  densitypxmz = vxZ[vecinopxmz].x / ncellsGPU;
  densitymxpy = vxZ[vecinomxpy].x / ncellsGPU;
  densitymxpz = vxZ[vecinomxpz].x / ncellsGPU;
  densitypypz = vxZ[vecinopypz].x / ncellsGPU;
  densitypymz = vxZ[vecinopymz].x / ncellsGPU;
  densitymypz = vxZ[vecinomypz].x / ncellsGPU;

  
  sXY += omega5 * 0.125 * ((density4 + density)  * vy  + (densitypxpy + density3)    * vy3)    * (vx  + vx4);
  sXY -= omega5 * 0.125 * ((density  + density1) * vy1 + (density3    + densitypxmy) * vypxmy) * (vx1 + vx);
  sXZ += omega5 * 0.125 * ((density5 + density)  * vz  + (densitypxpz + density3)    * vz3)    * (vx  + vx5);
  sXZ -= omega5 * 0.125 * ((density  + density0) * vz0 + (density3    + densitypxmz) * vzpxmz) * (vx0 + vx);
  sYX += omega5 * 0.125 * ((density3 + density)  * vx  + (densitypxpy + density4)    * vx4)    * (vy  + vy3);
  sYX -= omega5 * 0.125 * ((density  + density2) * vx2 + (density4    + densitymxpy) * vxmxpy) * (vy2 + vy);
  sYZ += omega5 * 0.125 * ((density5 + density)  * vz  + (densitypypz + density4)    * vz4)    * (vy  + vy5);
  sYZ -= omega5 * 0.125 * ((density  + density0) * vz0 + (density4    + densitypymz) * vzpymz) * (vy0 + vy);
  sZX += omega5 * 0.125 * ((density3 + density)  * vx  + (densitypxpz + density5)    * vx5)    * (vz  + vz3);
  sZX -= omega5 * 0.125 * ((density  + density2) * vx2 + (density5    + densitymxpz) * vxmxpz) * (vz2 + vz);
  sZY += omega5 * 0.125 * ((density4 + density)  * vy  + (densitypypz + density5)    * vy5)    * (vz  + vz4);
  sZY -= omega5 * 0.125 * ((density  + density1) * vy1 + (density5    + densitymypz) * vymypz) * (vz1 + vz);


  //STRESS TENSOR CONTRIBUTION
  sXX -= omega5 * (2. * shearviscosityGPU * invdxGPU * (vx3 - vx) + fact3GPU *
		   (invdxGPU * (vx3 - vx) + invdyGPU * (vy3 - vypxmy) + invdzGPU * (vz3 - vzpxmz)));
  sXX += omega5 * (2. * shearviscosityGPU * invdxGPU * (vx - vx2) + fact3GPU *
		   (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));
  sYY -= omega5 * (2. * shearviscosityGPU * invdyGPU * (vy4 - vy) + fact3GPU * 
		   (invdxGPU * (vx4 - vxmxpy) + invdyGPU * (vy4 - vy) + invdzGPU * (vz4 - vzpymz)));
  sYY += omega5 * (2. * shearviscosityGPU * invdyGPU * (vy - vy1) + fact3GPU *
		   (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));
  sZZ -= omega5 * (2. * shearviscosityGPU * invdzGPU * (vz5 - vz) + fact3GPU *
		   (invdxGPU * (vx5 - vxmxpz) + invdyGPU * (vy5 - vymypz) + invdzGPU * (vz5 -vz)));
  sZZ += omega5 * (2. * shearviscosityGPU * invdzGPU * (vz - vz0) + fact3GPU *
		   (invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0)));

  sXY -= omega5 * shearviscosityGPU * (invdyGPU * (vx4 - vx) + invdxGPU * (vy3 - vy));
  sXY += omega5 * shearviscosityGPU * (invdyGPU * (vx - vx1) + invdxGPU * (vypxmy - vy1));
  sXZ -= omega5 * shearviscosityGPU * (invdzGPU * (vx5 - vx) + invdxGPU * (vz3 - vz));
  sXZ += omega5 * shearviscosityGPU * (invdzGPU * (vx - vx0) + invdxGPU * (vzpxmz - vz0));
  sYX -= omega5 * shearviscosityGPU * (invdxGPU * (vy3 - vy) + invdyGPU * (vx4 - vx));
  sYX += omega5 * shearviscosityGPU * (invdxGPU * (vy - vy2) + invdyGPU * (vxmxpy - vx2));
  sYZ -= omega5 * shearviscosityGPU * (invdzGPU * (vy5 - vy) + invdyGPU * (vz4 - vz));
  sYZ += omega5 * shearviscosityGPU * (invdzGPU * (vy - vy0) + invdyGPU * (vzpymz - vz0));
  sZX -= omega5 * shearviscosityGPU * (invdxGPU * (vz3 - vz) + invdzGPU * (vx5 - vx));
  sZX += omega5 * shearviscosityGPU * (invdxGPU * (vz - vz2) + invdzGPU * (vxmxpz - vx2));
  sZY -= omega5 * shearviscosityGPU * (invdyGPU * (vz4 - vz) + invdzGPU * (vy5 - vy));
  sZY += omega5 * shearviscosityGPU * (invdyGPU * (vz - vz1) + invdzGPU * (vymypz - vy1));
  






  //Stress noise contribution
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  int n0;
  double fact1, fact2, fact4;

  fact1 = sqrt(omega2) * fact1GPU;
  fact2 = sqrt(omega2) * fact2GPU;
  fact4 = sqrt(omega2) * fact4GPU;

  n0 = 0;
  
  dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellsGPU] + d_rand[n0 + vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
  sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellsGPU] + d_rand[n0 + vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellsGPU] + d_rand[n0 + vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + i + ncellsGPU];
  sXY += fact4 * dnoise_sXY;
  sYX += fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + i + 2*ncellsGPU];
  sXZ += fact4 * dnoise_sXZ;
  sZX += fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + i + 4*ncellsGPU];
  sYZ += fact4 * dnoise_sYZ;
  sZY += fact4 * dnoise_sYZ;
  
  dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellsGPU] + d_rand[n0 + i + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
  sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_sYY = d_rand[n0 + i + 3*ncellsGPU] - dnoise_tr/3.;
  sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_sZZ = d_rand[n0 + i + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + vecino1 + ncellsGPU];
  sXY -= fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellsGPU];
  sXZ -= fact4 * dnoise_sXZ;
  
  dnoise_sXY = d_rand[n0 + vecino2 + ncellsGPU];
  sYX -= fact4 * dnoise_sXY;
  
  dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellsGPU];
  sYZ -= fact4 * dnoise_sYZ;
  
  dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellsGPU];
  sZX -= fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellsGPU];
  sZY -= fact4 * dnoise_sYZ;
  

  //Second random number
  n0 = ncellsGPU * 6;
  fact1 = sqrt(1-omega2) * fact1GPU;
  fact2 = sqrt(1-omega2) * fact2GPU;
  fact4 = sqrt(1-omega2) * fact4GPU;

  dnoise_tr = d_rand[n0 + vecino3] + d_rand[n0 + vecino3 + 3*ncellsGPU] + d_rand[n0 + vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + vecino3] - dnoise_tr/3.;
  sXX += fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino4] + d_rand[n0 + vecino4 + 3*ncellsGPU] + d_rand[n0 + vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[n0 + vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  sYY += fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_tr = d_rand[n0 + vecino5] + d_rand[n0 + vecino5 + 3*ncellsGPU] + d_rand[n0 + vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[n0 + vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ += fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + i + ncellsGPU];
  sXY += fact4 * dnoise_sXY;
  sYX += fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + i + 2*ncellsGPU];
  sXZ += fact4 * dnoise_sXZ;
  sZX += fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + i + 4*ncellsGPU];
  sYZ += fact4 * dnoise_sYZ;
  sZY += fact4 * dnoise_sYZ;
  
  dnoise_tr = d_rand[n0 + i] + d_rand[n0 + i + 3*ncellsGPU] + d_rand[n0 + i + 5*ncellsGPU];
  dnoise_sXX = d_rand[n0 + i] - dnoise_tr/3.;
  sXX -= fact1 * dnoise_sXX + fact2 * dnoise_tr;
  
  dnoise_sYY = d_rand[n0 + i + 3*ncellsGPU] - dnoise_tr/3.;
  sYY -= fact1 * dnoise_sYY + fact2 * dnoise_tr;
  
  dnoise_sZZ = d_rand[n0 + i + 5*ncellsGPU] - dnoise_tr/3.;
  sZZ -= fact1 * dnoise_sZZ + fact2 * dnoise_tr;
  
  dnoise_sXY = d_rand[n0 + vecino1 + ncellsGPU];
  sXY -= fact4 * dnoise_sXY;
  
  dnoise_sXZ = d_rand[n0 + vecino0 + 2*ncellsGPU];
  sXZ -= fact4 * dnoise_sXZ;
  
  dnoise_sXY = d_rand[n0 + vecino2 + ncellsGPU];
  sYX -= fact4 * dnoise_sXY;
  
  dnoise_sYZ = d_rand[n0 + vecino0 + 4*ncellsGPU];
  sYZ -= fact4 * dnoise_sYZ;
  
  dnoise_sXZ = d_rand[n0 + vecino2 + 2*ncellsGPU];
  sZX -= fact4 * dnoise_sXZ;
  
  dnoise_sYZ = d_rand[n0 + vecino1 + 4*ncellsGPU];
  sZY -= fact4 * dnoise_sYZ;

  









  px += -(invdxGPU * sXX + invdyGPU * sXY + invdzGPU * sXZ)*dtGPU;
  py += -(invdxGPU * sYX + invdyGPU * sYY + invdzGPU * sYZ)*dtGPU;
  pz += -(invdxGPU * sZX + invdyGPU * sZY + invdzGPU * sZZ)*dtGPU;



  dpxGPU[i] = px;
  dpyGPU[i] = py;
  dpzGPU[i] = pz;


}



















//STEP 8: solve vx^{n+1} and rho^{n+1}
__global__ void kernelDpSemiImplicitCompressibleParticles_4(const cufftDoubleComplex *vyZ,
							    double *densityGPU,
							    double *vxGPU,
							    double *vyGPU,
							    double *vzGPU,
							    const double *dpxGPU,
							    const double *dpyGPU,
							    const double *dpzGPU,
							    const double omega4){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega4 = 1.70710678118654752;
  //double omega4 = 0.292893218813452476;

  int vecino3, vecino4, vecino5; 
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  
  double density  = vyZ[i].x / ncellsGPU;
  double density3 = vyZ[vecino3].x / ncellsGPU;
  double density4 = vyZ[vecino4].x / ncellsGPU;
  double density5 = vyZ[vecino5].x / ncellsGPU;

  double pressure = pressure_GPU(density);
  double pressure3 = pressure_GPU(density3);
  double pressure4 = pressure_GPU(density4);
  double pressure5 = pressure_GPU(density5);
  
  

  //Explicit contribution
  double px = dpxGPU[i];
  double py = dpyGPU[i];
  double pz = dpzGPU[i];
  
  
  //Semi-implicit contribution
  double sXX = omega4 * (pressure3 - pressure) ;
  double sYY = omega4 * (pressure4 - pressure) ;
  double sZZ = omega4 * (pressure5 - pressure) ;


  px += -(invdxGPU * sXX)*dtGPU;
  py += -(invdyGPU * sYY)*dtGPU;
  pz += -(invdzGPU * sZZ)*dtGPU;

  vxGPU[i] = px * 2. / (density + density3);
  vyGPU[i] = py * 2. / (density + density4);
  vzGPU[i] = pz * 2. / (density + density5);


  densityGPU[i] = density;






}

























