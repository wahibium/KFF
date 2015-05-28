close all
clear all
clc

%% We make two simulations. One to calibrate, One to Measure.

for cal=(0:1:2)

clear LinearGrid

Nx=96;
Ny=512;
Nz=1;%% 3 layers to do a 2D simulation 

simul.dx=25e-9;% Cell size in the X direction
simul.dy=25e-9;% Cell size in the Y direction
simul.dz=50e-9;% Cell size in the Z direction
simul.pad=1;% Pad in the X and Y direction so that the size is divisible by 16

simul.x=Nx*simul.dx;%% Approximate total simulation size in the X direction
simul.y=Ny*simul.dy;%% Approximate total simulation size in the Y direction
simul.z=Nz*simul.dz;%% Approximate total simulation size in the Z direction
[LinearGrid,simul]=CreateConstantGrid(simul); %% Create a constant Grid

c=3e8;% Speed of light
dt=0.9*1/(c*sqrt(1/(simul.dx^2)+(1/simul.dy^2)+(1/simul.dz)^2)); % Timestep using a courant factor of 0.9
nsteps=4000;% Number of timesteps
LinearGrid.info.dt=dt; % Set the timestep
LinearGrid.info.tt=nsteps; %Set the number of timesteps
lambda= 1500e-9;
omegaTM=(2*pi*3e8)/lambda;

%% Termination to simulate TE or TM modes

%% TM Modes (Ez Hx Hy are supported)---> Terminate Z with PEC;
%%LinearGrid=SetAllBorders(LinearGrid,'000011',1,'perfectlayerzone','PEC');
%%LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PMC');
%% TE Modes (Hz Ex Ey are supported)---> Terminate Z with PMC;
LinearGrid=SetAllBorders(LinearGrid,'000001',1,'perfectlayerzone','PMC');
LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PEC');
nSiR=2.5;
%% Set Matarials


switch cal
    
    case 0 
        nSi=1;
    case 1 
        nSi=nSiR;
    case 2 
        nSi=100000; %Fake to have everything reflected
end
Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
LinearGrid=AddMat(LinearGrid,Air);


Si.name='Si'
Si.epsilon=nSi^2; %% Refractive index of oxide
LinearGrid=AddMat(LinearGrid,Si); % Defining oxide
d=lambda/(2);


%%

Sizone.name='Si';
Sizone.dx='end';
Sizone.dy='end';
Sizone.dz='end';
Sizone.z=1;
Sizone.y=1;
Sizone.x=Nx-20;
LinearGrid=AddMatZone2(LinearGrid,Sizone);





%% Add CPML
cpml.dx=10;
cpml.dy=10;
cpml.dz=10;
cpml.m=4;
cpml.xpos=1;
cpml.xneg=1;
cpml.ypos=1;
cpml.yneg=1;
cpml.zpos=0;
cpml.zneg=0;
cpml.amax=0.2000;
cpml.kmax=1;
cpml.smax=210000;
LinearGrid=AddCPML(LinearGrid,cpml);


%% Set sources.
source.dx=0;
source.dy=0;
source.dz='end';
source.x=12;
source.y=12;
source.z=1;
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/20;
%%source.Ez=1; %TM
source.Hz=1;%TE 

source.type='constant';
LinearGrid=DefineSpecialSource(LinearGrid,source);
%% Add Perfect Protector
%LinearGrid=AddPerfectConductor(LinearGrid,1,source.y+5,source.z,source.x,5,'end',{'Hz'});


B=4*omegaTM/(2*pi); %desired bandwidth
output.x=1;
output.y=1;
output.z=1;
output.dx='end';

output.dy='end';
output.dz=0;
output.field=[{'Ez','Hz'}];
output.deltaT=10;%floor(1/(2*B*dt));%% number of timesteps;
output.name='inclined_slab';
output.foutstart=omegaTM*0.8/(2*pi);
output.foutstop=omegaTM*1.05/(2*pi);%% So we get all the frequencies.
output.field=[{'Hz'},{'Ex'},{'Ey'}];
LinearGrid=AddOutput(LinearGrid,output);

%% Where to simulate

SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.SimulatorExec='/home/pwahl/MyCode/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/InclinedSlab'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);
%% Copy Back

 %system(['scp tullibardine:' SimulInfoM.SimulFileRemote 'inclined_slab' ' ' SimulInfo.outfile]);
  
topology.field='topology';
topology.x='all';
topology.y='all';
topology.z=1;
topology.data=0;
topology.gridfield='Hz'

 


full=topology
full.x='all';
full.y='all';
full.z=1;
full.field='Hz_abs'
full.data=frequency2index([SimulInfo.outfile 'inclined_slabFFT'],omegaTM/(2*pi));
PlotData(full,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);



line=topology;
line.x=source.x;
line.y='all';
line.z=1;
line.noplot=1;
line.field='Hz_abs';
line.data=frequency2index([SimulInfo.outfile 'inclined_slabFFT'],omegaTM/(2*pi));
[data(cal+1,:),pos]=PlotData(line,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);
line.field='Hz_phase'
[phase(cal+1,:),pos]=PlotData(line,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);



end


[topologydata,posdata]=PlotData(topology,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);



data=data.*exp(j*phase);

a=(Sizone.x-source.x)*simul.dx;%%Distance between source and Slab
b=pos.x-pos.x(source.y);

r=abs(data(2,:)-data(1,:))./abs(data(3,:)-data(1,:));


theta=atan(b/(2*a));
%% Theory: Fresnel Law for P polirized field

nAir=1

n1=nAir;
n2=nSiR;

rth=abs((n1*(sqrt(1-(n1/n2*sin(theta)).^2)-n2*cos(theta)))./((n1*(sqrt(1-(n1/n2*sin(theta)).^2)+n2*cos(theta)))));
figure

theta=theta*180/pi;
hold on
 plot(theta,r,'y*')
plot(theta,rth,'b')
hold off




