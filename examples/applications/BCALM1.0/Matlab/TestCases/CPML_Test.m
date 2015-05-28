%system('cd ../../CUDA_code/; make clean ; make;')
%system('cd ../../CUDA_code/; make;')
addpath(genpath('../Matlab_utils/'))
clear all
close all
clc



simul.x=12.8e-6;%% Approximate total simulation size in the X direction
simul.y=12.8e-6;%% Approximate total simulation size in the Y direction
simul.z=12.8e-6;%% Approximate total simulation size in the Z direction

simul.dx=200e-9;% Cell size in the X direction
simul.dy=200e-9;% Cell size in the Y direction
simul.dz=200e-9;% Cell size in the Z direction
simul.pad=1;% Pad in the X and Y direction so that the size is divisible by 16
[LinearGrid,simul]=CreateConstantGrid(simul); %% Create a constant Grid

Nx=LinearGrid.info.xx; 
Ny=LinearGrid.info.yy;
Nz=LinearGrid.info.zz;




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
cpml.dx=10;
cpml.dy=10;
cpml.dz=10;
cpml.m=4;
cpml.xpos=0;
cpml.xneg=0;
cpml.ypos=0;
cpml.yneg=0;
cpml.zpos=1;
cpml.zneg=1;
cpml.amax=0.2000;
cpml.kmax=1;
cpml.smax=210000;
LinearGrid=AddCPML(LinearGrid,cpml);

%% Termination
LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PMC');
LinearGrid=SetAllBorders(LinearGrid,'110000',1,'perfectlayerzone','PEC');

%% Set sources.
lambda= 2000e-9;
omegaTM= (2*pi*3e8)/lambda;


source.x=1
source.y=1;
source.z=Nz/2;
source.dx='end';
source.dy='end';
source.dz=0;
source.omega=omegaTM;
source.mut=1;
source.sigmat=nsteps/10;
source.Ex=1;
source.type='constant';
source.n=1;
source.w0=1.5*lambda;

LinearGrid=DefineSpecialSource(LinearGrid,source);





%% Define New outputs

output.x=1
output.y=Ny/2;
output.z=1;
output.dx='end';
output.dy=1;
output.dz='end';
output.deltaT=4;%% number of timesteps;
output.name='cpml_test';
output.foutstart=0.9*omegaTM/(2*pi);
output.foutstop=1.8*omegaTM/(2*pi);%% So we get all the frequencies.
output.field=[{'Ex'},{'Hx'}];
LinearGrid=AddOutput(LinearGrid,output);
output2=output;
output2.name='cpml_test2';
output2.y=1;
output2.dy='end';
output2.z=Nz/2;
output2.dz=0;
LinearGrid=AddOutput(LinearGrid,output2);




%% Launch simulation
%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
%SimulInfo.SimulatorExec='/home/pwahl/MyCode/obj/fdtd';
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/CPMLTEST'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);

%% Plot

%    load('debugzone3.txt');
%    load('debugzone4.txt');
%  plot(debugzone3(:,2),debugzone3(:,1),'*')
%  hold all
%  plot(debugzone3(:,2),debugzone3(:,3),'*')
% 
%   plot(debugzone4(:,2)+0.5,debugzone4(:,1),'*')
%  plot(debugzone4(:,2)+0.5,debugzone4(:,3),'*')
%  
%  dataT2=permute(dataT,[1,3,2,4]);
 
 %MyLittlePlotOut(dataT2)  ;
 
  
% SimulInfo.WorkPlace = '../Work/';
%   
% 
full.x='all';
full.y=1;
full.z='all';
full.plottype='normal';
full.field='Ex_abs';
full.noplot=0;
full.gridfield='Ex'
full.data=frequency2index([SimulInfo.outfile 'cpml_testFFT'],omegaTM/(2*pi));
PlotData(full,LinearGrid,[SimulInfo.outfile 'cpml_testFFT']);
shading interp





full3.x=32;
full3.y='all';
full3.z=1;
full3.plottype='normal';
full3.field = 'Ex_abs';
full3.noplot=0;
full3.gridfield='Ex'
full3.data=frequency2index([SimulInfo.outfile 'cpml_testFFT'],omegaTM/(2*pi));
[xx,pos] = PlotData(full3,LinearGrid,[SimulInfo.outfile 'cpml_test2FFT']);
shading interp
debugout(end,:)
