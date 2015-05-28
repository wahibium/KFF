close all
clear all
clc

%% 2D simulation 

Nx=128;
Ny=128;
Nz=2;%% 3 layers to do a 2D simulation 

SIZE=[Nx Ny Nz];


simul.dx=2e-9;% Coarse Grid;
simul.dy=2e-9;
simul.dz=2e-9;

simul.x=Nx*simul.dx
simul.y=Ny*simul.dy
simul.z=Nz*simul.dz


simul.pad=1;
[LinearGrid,simul]=CreateConstantGrid(simul);


c=3e8;
dt=0.9*1/(c*sqrt(1/(simul.dx^2)+(1/simul.dy^2)+(1/simul.dz)^2)); % Timestep using a courant factor of 0.9
nsteps=100000;
lambda= 1500e-9;
omegaTM=(2*pi*3e8)/lambda;
LinearGrid.info.dt=dt; % Set the timestep
LinearGrid.info.tt=nsteps; %Set the number of timesteps

%% Termination to simulate TE or TM modes

%% TM Modes (Ez Hx Hy are supported)---> Terminate Z with PEC;
%%LinearGrid=SetAllBorders(LinearGrid,'000011',1,'perfectlayerzone','PEC');
%%LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PMC');
%% TE Modes (Hz Ex Ey are supported)---> Terminate Z with PMC;
LinearGrid=SetAllBorders(LinearGrid,'000011',1,'perfectlayerzone','PMC');
LinearGrid=SetAllBorders(LinearGrid,'111100',1,'perfectlayerzone','PEC');

%% Set Matarials
wpm=[1.014076882e16,6.861574653e15,1.832181311e15];
wm=[0,2.046174861e10,8.225155873e14];
gammam=0*[0,4.390839341e14,3.52118567e15];

lambdatest=(1200e-9:10e-9:2000e-9);
omegatest=(2*pi*3e8)./lambdatest;
E=1.05457148e-34*omegatest/1.60217646e-19;
epsr=ones(1,length(omegatest));
eps0=8.854187817E-12;
mu0=4*pi*1e-7;
sumgamma=0;
 for m=(1:length(wpm))
    
   epsr=epsr+ (wpm(m))^2./(wm(m)^2-omegatest.^2+i*omegatest*gammam(m));

		alpha(m) = (2-wm(m)^2*dt^2)/(1+dt*gammam(m)/2);
		zeta(m)  = (dt*gammam(m)/2 - 1) / (dt*gammam(m)/2 + 1);
		gamma(m) = (wpm(m)*wpm(m)*eps0*dt*dt) / (dt*gammam(m)/2 + 1);
        sumgamma=gamma(m)+sumgamma;
     
 end

 C1=(sumgamma/2)/(2*eps0+sumgamma/2);
 C2=2*eps0/(2*eps0+sumgamma/2);
 C3=2*dt/(2*eps0+sumgamma/2);

figure
hold on
plot(lambdatest,real(epsr),'k*');
plot(lambdatest,imag(epsr),'r*')

Au.poles=[wpm',wm',gammam'];
Au.name='Au';
Au.epsilon=1;
LinearGrid=AddMat(LinearGrid,Au);

Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
LinearGrid=AddMat(LinearGrid,Air);



%%




Auzone.name='Au'
Auzone.y=1
Auzone.dy='end';
Auzone.x=1;
Auzone.dx='end';
Auzone.z=1;
Auzone.dz='end';
LinearGrid=AddMatZone2(LinearGrid,Auzone);








%% Add CPML
%% Section 6: Define CPML's (Absorbing boundary condtions)
cpml.dx=10;% Width of the CPML in the x direction (#cells)
cpml.dy=10;% Width of the CPML in the y direction (#cells)
cpml.dz=0;% Width of the CPML in the z direction (#cells)
cpml.xpos=1;% Add a CPML on the right X direction
cpml.xneg=1;% Add a CPML on the left X direction
cpml.ypos=1;% Add a CPML on the right Y direction
cpml.yneg=1;% Add a CPML on the left Y direction
cpml.zpos=0;% Add a CPML on the right Z direction
cpml.zneg=0;% Add a CPML on the right Z direction

cpml.m=4; %CPML specific don't really need to change
cpml.amax=0.2000; %CPML specific don't really need to change
cpml.kmax=1;%CPML specific don't really need to change
cpml.smax=210000;%CPML specific don't really need to change
LinearGrid=AddCPML(LinearGrid,cpml); %% Add the CPML

%% Set sources.
source.dx=0;
source.dy='end';
source.dz='end';
source.x=15;
source.y=1;
source.z=1;
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/10;
source.Hz=1;%TE 

source.type='constant';
LinearGrid=DefineSpecialSource(LinearGrid,source);
%% Outputs


B=2*omegaTM/(2*pi); %desired bandwidth
output.x=1;
output.y=1;
output.z=1;
output.dx='end';
output.dy='end';
output.dz=0;
output.deltaT=floor(1/(2*B*dt));%% number of timesteps;
output.name='TestBulk2D';
output.foutstart=0.8*omegaTM/(2*pi);
output.foutstop=1.1*omegaTM/(2*pi);%% So we get all the frequencies.
output.field=[{'Hz'}];
LinearGrid=AddOutput(LinearGrid,output);


%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/DielectricCube'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);


%% Plot Results
topology.field='topology';
topology.x='all';
topology.y='all';
topology.z=1;
topology.data=0;
topology.gridfield='Hz'
%topology.gridfield='Ez'
[topologydata,posdata]=PlotData(topology,LinearGrid,[SimulInfo.outfile 'TestBulk2DFFT']);
shading interp


full=topology
full.x='all';
full.y='all';
full.z=1;
full.plottype='normal'
full.field='Hz_real'
full.plottype='linear'
full.noplot=0;
full.data=frequency2index([SimulInfo.outfile 'TestBulk2DFFT'],omegaTM/(2*pi));
PlotData(full,LinearGrid,[SimulInfo.outfile 'TestBulk2DFFT']);
shading interp

line=full;
line.y=Ny/2;
PlotData(line,LinearGrid,[SimulInfo.outfile 'TestBulk2DFFT']);

sourcecheck=line;
sourcecheck.field='Hz_abs'
sourcecheck.data='all'
sourcecheck.x=source.x
sourcecheck.y=source.y
sourcecheck.z=source.z
PlotData(sourcecheck,LinearGrid,[SimulInfo.outfile 'TestBulk2DFFT']);



test=topology
test.field='Hz_abs'
test.x=3*Nx/4;
test.y='all'
test.z=1;
test.plottype='normal'
test.noplot=0;
test.data=frequency2index([SimulInfo.outfile 'TestBulk2DFFT'],omegaTM/(2*pi));
[testdata,pos]=PlotData(test,LinearGrid,[SimulInfo.outfile 'TestBulk2DFFT']);



%
 dataT=Extract([SimulInfo.outfile 'TestBulk2D'], 'Hz',4);
 MyLittlePlotOut(dataT)
%
sourcecheck.field='Hz'
sourcecheck.data='all'
sourcecheck.plottype='linear'
[testdata,pos]=PlotData(sourcecheck,LinearGrid,[SimulInfo.outfile 'TestBulk2D']);







