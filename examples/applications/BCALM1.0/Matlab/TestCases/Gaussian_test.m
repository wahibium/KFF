clear all
close all
clc

Nx=256;
Ny=128;
Nz=128;

simul.dx=200e-9;% Cell size in the X direction
simul.dy=200e-9;% Cell size in the Y direction
simul.dz=200e-9;% Cell size in the Z direction
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


%% Materials
Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
LinearGrid=AddMat(LinearGrid,Air);

%% AddCPML
%% Termination

cpml.dx=10;
cpml.dy=10;
cpml.dz=10;
cpml.m=4;
cpml.xpos=1;
cpml.xneg=1;
cpml.ypos=1;
cpml.yneg=1;
cpml.zpos=1;
cpml.zneg=1;
cpml.amax=0.2000;
cpml.kmax=1;
cpml.smax=210000;



LinearGrid=AddCPML(LinearGrid,cpml);
%% Set sources.
lambda= 2000e-9;
omegaTM= (2*pi*3e8)/lambda;


source.x=Nx/2;
source.y=Ny/2;
source.z=2*Nz/4;
source.dx=0;
source.dy=0;
source.dz='end';
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/10;
source.Ey=1;
%source.facex=15; %% sweeps over all yz.
%source.facey=12;
source.facez=15;
source.type='gaussianbeam';
source.n=1;
source.w0=1.5*lambda;
source.phi=pi/15;
source.tetha=0;

LinearGrid=DefineSpecialSource(LinearGrid,source);





%% Define New outputs

output.x=1
output.y=Ny/2;
output.z=1;
output.dx='end';
output.dy=1;
output.dz='end';
output.deltaT=4;%% number of timesteps;
output.name='Gaussian_test';
output.foutstart=0.9*omegaTM/(2*pi);
output.foutstop=1.8*omegaTM/(2*pi);%% So we get all the frequencies.
output.field=[{'Ey'}];
LinearGrid=AddOutput(LinearGrid,output);
output2=output;
output2.name='Gaussian_test2';
output2.y=1;
output2.dy='end';
output2.z=Nz/2;
output2.dz=0;
LinearGrid=AddOutput(LinearGrid,output2);




%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.SimulatorExec='/home/pwahl/MyCode/obj/fdtd'; %% Place where the fdtd program is located

SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/GaussianTEST'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);
%% Plot


%  system(['scp tullibardine:' SimulInfoM.SimulFileRemote 'Gaussian_test' ' ' SimulInfoM.WorkPlace]);
%  dataT=Extract([SimulInfoM.WorkPlace 'Gaussian_test'], 'Ey',4);
%  MyLittlePlotOut(dataT)  

full.x='all';
full.y=1;
full.z='all';
full.plottype='normal';
full.field='Ey_real';
full.noplot=0;
full.gridfield='Ey'
full.data=frequency2index([SimulInfo.outfile 'Gaussian_testFFT'],omegaTM/(2*pi));
PlotData(full,LinearGrid,[SimulInfo.outfile 'Gaussian_testFFT']);
shading interp

full.field='topology'

PlotData(full,LinearGrid,[SimulInfo.outfile 'Gaussian_testFFT']);
full2=full;
full2.y=Ny/2;
full2.z=1;
full2.noplot=1;

full2.field='Ey_abs'
[data,pos]=PlotData(full2,LinearGrid,[SimulInfo.outfile 'Gaussian_test2FFT']);

full2.field='Ey_real'
full2.noplot=0;
full2.y='all';
PlotData(full2,LinearGrid,[SimulInfo.outfile 'Gaussian_test2FFT']);
shading interp
figure

plot(pos.x,data);
hold on
[FF,Idx] = max(data);
mu_x = pos.x(Idx);
plot(pos.x,FF*exp(-(pos.x-mu_x).^2/source.w0^2),'*y')
hold off




