close all
clear all
clc
Nx=128;
Ny=64;
Nz=2;%% 3 layers to do a 2D simulation 


simul.dx=12.5e-9;% Cell size in the X direction
simul.dy=12.5e-9;% Cell size in the Y direction
simul.dz=50e-9;% Cell size in the Z direction
simul.pad=1;% Pad in the X and Y direction so that the size is divisible by 16

simul.x=Nx*simul.dx;%% Approximate total simulation size in the X direction
simul.y=Ny*simul.dy;%% Approximate total simulation size in the Y direction
simul.z=Nz*simul.dz;%% Approximate total simulation size in the Z direction
[LinearGrid,simul]=CreateConstantGrid(simul); %% Create a constant Grid

c=3e8;% Speed of light
dt=0.9*1/(c*sqrt(1/(simul.dx^2)+(1/simul.dy^2)+(1/simul.dz)^2)); % Timestep using a courant factor of 0.9
nsteps=5000;% Number of timesteps
LinearGrid.info.dt=dt; % Set the timestep
LinearGrid.info.tt=nsteps; %Set the number of timesteps

lambda= 1000e-9;
omegaTM=(2*pi*3e8)/lambda;




%% Termination to simulate TE or TM modes

%% TM Modes (Ez Hx Hy are supported)---> Terminate Z with PEC;
%%LinearGrid=SetAllBorders(LinearGrid,'000011',1,'perfectlayerzone','PEC');
%%LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PMC');
%% TE Modes (Hz Ex Ey are supported)---> Terminate Z with PMC;
LinearGrid=SetAllBorders(LinearGrid,'000011',1,'perfectlayerzone','PMC');
LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PEC');

%% Set Matarials
nSi=2;

Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
LinearGrid=AddMat(LinearGrid,Air);


Si.name='Si'
Si.epsilon=nSi^2; %% Refractive index of oxide
LinearGrid=AddMat(LinearGrid,Si); % Defining oxide
d=lambda/(2);
d=lambda/((4*2*nSi));
d=lambda/2;

Sizone.name='Si',
Sizone.dx=floor((d/simul.dx))-1;
Sizone.dy='end';
Sizone.dz='end';
Sizone.x=Nx/2;
Sizone.y=1;
Sizone.z=1;
LinearGrid=AddMatZone2(LinearGrid,Sizone);




%% Add CPML
%% Termination

cpml.dx=10;
cpml.dy=10;
cpml.dz=10;
cpml.m=4;
cpml.xpos=1;
cpml.xneg=1;
cpml.ypos=0;
cpml.yneg=0;
cpml.zpos=0;
cpml.zneg=0;
cpml.amax=0.2000;
cpml.kmax=1;
cpml.smax=210000;



LinearGrid=AddCPML(LinearGrid,cpml);

%% Set sources.
source.dx=0;
source.dy='end';
source.dz='end';
source.x=15;
source.y=1;
source.z=1;
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/20;
%%source.Ez=1; %TM
source.Hz=1;%TE 

source.type='constant';
LinearGrid=DefineSpecialSource(LinearGrid,source);


B=4*omegaTM/(2*pi); %desired bandwidth
output.x=1;
output.y=1;
output.z=1;
output.dx='end';
output.dy='end';
output.dz=0;
output.field=[{'Ez','Hz'}];
output.deltaT=10;%floor(1/(2*B*dt));%% number of timesteps;
output.name='normal_slab';
output.foutstart=omegaTM*0.5/(2*pi);
output.foutstop=omegaTM*1.5/(2*pi);%% So we get all the frequencies.
output.field=[{'Hz'},{'Ex'},{'Ey'}];
LinearGrid=AddOutput(LinearGrid,output);

%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/normalSlab'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);

%% Plot
 
 
topology.field='topology';
topology.x='all';
topology.y='all';
topology.z=1;
topology.data=0;
topology.gridfield='Hz'
PlotData(topology,LinearGrid,[SimulInfo.outfile 'normal_slabFFT']);

topology.field='Hz_abs';
topology.x='all'
topology.y=Ny/2;
topology.z=1;
topology.data=frequency2index([SimulInfo.outfile 'normal_slabFFT'],omegaTM/(2*pi));
topology.gridfield='Hz'
PlotData(topology,LinearGrid,[SimulInfo.outfile 'normal_slabFFT']);

source=topology
source.field='Hz_abs'
source.noplot=1;
source.y=Ny/2;
source.x='all'
source.z=1;


PlotData(source,LinearGrid,[SimulInfo.outfile 'normal_slabFFT']);


source.x='all';
source.y='all';
source.z=1;
source.data=frequency2index([SimulInfo.outfile 'normal_slabFFT'],omegaTM/(2*pi));
PlotData(source,LinearGrid,[SimulInfo.outfile 'normal_slabFFT']);

source.x=Nx/2;
source.y=Ny/2;
source.z=1;
source.field='Hz_abs';
source.data='all';
[data,pos]=PlotData(source,LinearGrid,[SimulInfo.outfile 'normal_slabFFT']);





%% Theory

%% Getting data out.

source.x='all'
source.noplot=1;
[dataTH,posTH]=PlotData(source,LinearGrid,[SimulInfo.outfile 'normal_slabFFT']);
[a,b]=size(dataTH);

istart=11;
istop=Sizone.x-1;

for (f=1:b)
  temp=dataTH(:,f);
  
  MIN=min(temp(istart:istop));
  MAX=max(temp(istart:istop));

  
  gammafound(f)=(MAX-MIN)/((MAX+MIN));
 
end


n1=1;
n2=nSi;
n3=1;
c=3e8;
lambda2=c./pos.x;
beta2=2*pi*nSi./(lambda2);

GAMMA=((n2-n1)*(n3+n2)+(n2+n1)*(n3-n2)*exp(-2*j*beta2*d))./((n2+n1)*(n3+n2)+(n2-n1)*(n3-n2)*exp(-2*j*beta2*d));
GAMMA=abs(GAMMA);
figure
hold on
plot(pos.x,GAMMA)
plot(pos.x,gammafound,'y*')





