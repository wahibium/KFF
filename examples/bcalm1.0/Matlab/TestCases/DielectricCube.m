addpath(genpath('../Matlab_utils/'))
clear all
close all
clc

%% Section 1: Create grid

simul.x=12.8e-6;%% Approximate total simulation size in the X direction
simul.y=12.8e-6;%% Approximate total simulation size in the Y direction
simul.z=12.8e-6;%% Approximate total simulation size in the Z direction

simul.dx=100e-9;% Cell size in the X direction
simul.dy=100e-9;% Cell size in the Y direction
simul.dz=100e-9;% Cell size in the Z direction
simul.pad=1;% Pad in the X and Y direction so that the size is divisible by 16
[Grid,simul]=CreateConstantGrid(simul); %% Create a constant Grid

Nx=Grid.info.xx; 
Ny=Grid.info.yy;
Nz=Grid.info.zz;

%% Section 2: Defining number of timesteps and stepsize
c=3e8;% Speed of light
dt=0.9*1/(c*sqrt(1/(simul.dx^2)+(1/simul.dy^2)+(1/simul.dz)^2)); % Timestep using a courant factor of 0.9
nsteps=4000;% Number of timesteps
Grid.info.dt=dt; % Set the timestep
Grid.info.tt=nsteps; %Set the number of timesteps

%% Wavelength
lambda= 1.5e-6; % Wavelength 
omega= (2*pi*3e8)/lambda; % Angular Frequency
%% Section 3: Defining materials

Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
Grid=AddMat(Grid,Air);

Glass=Air;
Glass.name='Glass';
Glass.ambient=0;
Glass.epsilon=1.4^2;
Grid=AddMat(Grid,Glass);

%% Section 4: Add structures
Sizone.x=Nx/2-10;%Material zone start in (# cells)
Sizone.y=Ny/2-10;%Material zone start in Y  (# cells)
Sizone.z=Nz/2-10;%Material zone start in Z (# cells)
Sizone.dx=20; % With of the material zone in the X direction (# cells)
Sizone.dy=20; % With of the material zone in the Y direction (# cells)
Sizone.dz=20; % With of the material zone in the Z direction (# cells)
Sizone.name='Glass'; % Name of the Material  
Grid=AddMatZone2(Grid,Sizone); % Adding the Material to the Grid


%% Section 5: Add Sources Plane Wave or Gaussian Beam

% Plane wave

sourcePW.x=1; %% Plane wave exitation zone starts at X=1
sourcePW.y=1; %% Plane wave exitation zone starts at Y=1
sourcePW.z=20; %% %% Plane wave exitation zone starts at Z=20
sourcePW.dx='end';  %% Plane wave exitation zone covers the whole simulation area in the X direction
sourcePW.dy='end'; %% Plane wave exitation zone covers the whole simulation area in the Y direction
sourcePW.dz=0; %% Plane wave exitation zone is one cell thick in the Z direction.
sourcePW.type='constant'; %% Set the type to constant= Plane Wave
sourcePW.omega=omega; % Angular frequency of the exitation in (Hz)
sourcePW.mut=nsteps/4; % Mu of the temporal gaussian envelope (in timesteps)
sourcePW.sigmat=nsteps/10; % Sigma of the temporal gaussian envelope (in timesteps)
sourcePW.Ey=1; %Amplitude of the exitation of the Ey field;
Grid=DefineSpecialSource(Grid,sourcePW); % Add source.

% Gaussian Beam

% source.x=Nx/2;% Pos of the waist of the Gaussian Beam  (# cells)
% source.y=Ny/2;% Pos of the waist of the Gaussian Beam  (# cells)
% source.z=Nz/2;% Pos of the waist of the Gaussian Beam  (# cells)
% source.dx=0; %% Insignificant when defining a Gaussian Beam
% source.dy=0; %% Insignificant when defining a Gaussian Beam
% source.dz=0; %% Insignificant when defining a Gaussian Beam
% source.omega=omega; % Angular frequency of the exitation in (Hz)
% source.mut=nsteps/4; % Mu of the temporal gaussian envelope (in timesteps)
% source.sigmat=nsteps/10; % Sigma of the temporal gaussian envelope (in timesteps)
% source.Ey=1; %Amplitude of the exitation of the Ey field;
% %Gaussian Beam Specific parameters
% source.type='gaussianbeam'; %% Set the type to gaussianbeam
% source.facez=15; %%  From wich face are we exiting 
% source.n=1; % Index of the material between the source and the waist
% source.w0=1.5*lambda; % Size of the waist of the gaussian beam
% source.phi=0; %Angle Kvector of the Gaussian beam to and the Z axis
% source.tetha=0; % Angle between the projection of the K vector of the gaussian beam on the XY plane and the X axis
% Grid=DefineSpecialSource(Grid,source); % Add source.

%% Section 6: Define CPML's (Absorbing boundary condtions)
cpml.dx=10;% Width of the CPML in the x direction (#cells)
cpml.dy=10;% Width of the CPML in the y direction (#cells)
cpml.dz=10;% Width of the CPML in the z direction (#cells)
cpml.xpos=1;% Add a CPML on the right X direction
cpml.xneg=1;% Add a CPML on the left X direction
cpml.ypos=1;% Add a CPML on the right Y direction
cpml.yneg=1;% Add a CPML on the left Y direction
cpml.zpos=1;% Add a CPML on the right Z direction
cpml.zneg=1;% Add a CPML on the right Z direction

cpml.m=4; %CPML specific don't really need to change
cpml.amax=0.2000; %CPML specific don't really need to change
cpml.kmax=1;%CPML specific don't really need to change
cpml.smax=210000;%CPML specific don't really need to change
Grid=AddCPML(Grid,cpml); %% Add the CPML


%% Section 6: Outputs
output.x=1 % Start of the outputzone in the X direction (# cells)
output.y=Ny/2;% Start of the outputzone in the Y direction (# cells)
output.z=1;% Start of the outputzone in the Z direction (# cells)
output.dx='end'; % Width of the output zone in the X direction 'end=until the end of the simulation'
output.dy=0; % Width of the outputzone in the Y direction
output.dz='end';% Width of the output zone in the Z direction 'end=until the end of the simulation'
%output.deltaT=4;%% number of timesteps; (If not set automatically
%calculated to accomodate the required bandwidth)
output.foutstart=0.9*omega/(2*pi);%Minimum of the frequency range to be outputted(Hz)
output.foutstop=1.8*omega/(2*pi);%% Maximum of the frequency range to be outputted(Hz)
output.deltafmin=omega/(100*2*pi)%% Mininum frequency resolution
output.field=[{'Ex'},{'Ey'},{'Ez'},{'Hx'},{'Hy'},{'Hz'}];% Fields to be outputted
output.name='Example_Cross_Section'; % Name of the file 
Grid=AddOutput(Grid,output); %% Add the output

%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.SimulatorExec='/home/pwahl/MyCode/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/DielectricCube'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(Grid, SimulInfo);

%% Section 8: Post-processing

omegaidx = frequency2index([SimulInfo.WorkPlace 'Example_Cross_SectionFFT'],omega/(2*pi)) % Get the frequency bin the frequency of interest
full.x='all'; % Plot all the x points present in the output
full.y=1; % Plot the first y point present in the output 
% We only outputted one y coordinate so full.y=1 referes to that
% particular coordinate in the ouput file not in the grid itself.
full.z='all';% Output all the coordinates in the Z direction
full.plottype='linear'; % Linear type of plot ( aka not logaritmic)
full.gridfield='Ey'; %% Position of the Y fields is used
full.field='topology' %% Plot the topology of the structure within the plot
full.data=1% For topology plots data is field is irrelevant
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
full.data=omegaidx; %% Frequency bin used ( we only output 1 frequency)
full.field='Ey_real'%% Plot the real part of the Ey field
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
shading interp
full.field='Hx_abs'%% Plot the abselolute value of the Hx
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
shading interp
full.x=Nx/2;
full.z=Nz/2;
full.data='all';
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
full.x='all'
full.y=1
full.z='all'
full.data=omegaidx
full.field='Hx_phase'%%Plot the phase Hx
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
shading interp

full.field='ME2_real' %% PLots the ModE square of the fields
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
shading interp

full.field='Pz_abs' %% PLots the Pz
[data,pos]=PlotData(full,Grid,[SimulInfo.WorkPlace 'Example_Cross_SectionFFT']); %% Plot and get data
shading interp
