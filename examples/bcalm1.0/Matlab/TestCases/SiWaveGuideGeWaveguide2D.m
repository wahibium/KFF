close all
clear all
clc
%%
Siguide.width=400e-9;
Geguide.width=2000e-9;
Geguide.height=200e-9;

%%


Nz=2;

simul.dx=15e-9;% Cell size in the X direction
simul.dy=15e-9;% Cell size in the Y direction
simul.dz=25e-9;% Cell size in the Z direction
simul.pad=1;% Pad in the X and Y direction so that the size is divisible by 16

simul.x=5e-6;%% Approximate total simulation size in the X direction
simul.y=3e-6;%% Approximate total simulation size in the Y direction
simul.z=Nz*simul.dz;%% Approximate total simulation size in the Z direction
[LinearGrid,simul]=CreateConstantGrid(simul); %% Create a constant Grid

Nx=LinearGrid.info.xx; %Nx
Ny=LinearGrid.info.yy; %Ny



c=3e8;% Speed of light
dt=0.9*1/(c*sqrt(1/(simul.dx^2)+(1/simul.dy^2)+(1/simul.dz)^2)); % Timestep using a courant factor of 0.9
nsteps=20000;% Number of timesteps
LinearGrid.info.dt=dt; % Set the timestep
LinearGrid.info.tt=nsteps; %Set the number of timesteps

lambda=1700e-9;
omegaTM=(2*pi*3e8)/lambda;

LinearGrid.calibrate=0;
LinearGrid.recomputesource=1;

%% Termination to simulate TE or TM modes

%% TM Modes (Ez Hx Hy are supported)---> Terminate Z with PEC;
%%LinearGrid=SetAllBorders(LinearGrid,'000011',1,'perfectlayerzone','PEC');
%%LinearGrid=SetAllBorders(LinearGrid,'001100',1,'perfectlayerzone','PMC');
%% TE Modes (Hz Ex Ey are supported)---> Terminate Z with PMC;
LinearGrid=SetAllBorders(LinearGrid,'000001',1,'perfectlayerzone','PMC');

%% Set Matarials

Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
LinearGrid=AddMat(LinearGrid,Air);

Si.name='Si';
Si.epsilon=3.5^2;
Si.sigma=0;
LinearGrid=AddMat(LinearGrid,Si);


Ge.name='Ge';
Ge.epsilon=4^2;
Ge.sigma=0;
LinearGrid=AddMat(LinearGrid,Ge);

%% Add the materials zones


SiGuideZone.x=1;
SiGuideZone.dx='end';
SiGuideZone.dy=floor(Siguide.width/simul.dy)-1;
SiGuideZone.y=floor(Ny/3-SiGuideZone.dy/2);
SiGuideZone.z=1;
SiGuideZone.calibrate=0;
SiGuideZone.dz='end';
SiGuideZone.name='Si';
LinearGrid=AddMatZone(LinearGrid,SiGuideZone);

GeGuideZone=SiGuideZone;
GeGuideZone.calibrate=1;
GeGuideZone.name='Ge';
GeGuideZone.dx=floor(Geguide.width/simul.dy)-1;
GeGuideZone.x=floor(Nx/2-GeGuideZone.dx/2);
GeGuideZone.y=SiGuideZone.y+SiGuideZone.dy+1;
GeGuideZone.dy=floor(Geguide.height/simul.dy)-1;
LinearGrid=AddMatZone(LinearGrid,GeGuideZone);



%% Add CPML
cpml.dx=10;
cpml.dy=10;
cpml.dz=0;
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
source.dy=SiGuideZone.dy;
source.dz='end';
source.x=25;
source.y=SiGuideZone.y;
source.z=1;
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/10;
source.Hz=1;%TE 
source.type='constant';
if LinearGrid.recomputesource
LinearGrid=DefineSpecialSource(LinearGrid,source);
else
    source.dy=0;
    p=load('StephSource');
    msource=p.datasource;
    for y=1:length(msource)
        source.y=y;
        source.Hz=msource(y);
        LinearGrid=DefineSpecialSource(LinearGrid,source);
    end
end
    
    
%% Define the outputs

B=2.5*omegaTM/(2*pi); %desired bandwidth
output.x=1;
output.y=1;
output.z=1;
output.calibrate=1;
output.dx='end';
output.dy='end';
output.dz=0;
output.deltaT=floor(1/(2*B*dt));%% number of timesteps;
output.name='Si2GeWaveguideOut';
output.foutstart=omegaTM*0.9/(2*pi);
output.foutstop=omegaTM*1.1/(2*pi);
output.deltafmin=omegaTM/(2*pi*400);
output.field=[{'Ex'},{'Ey'},{'Ez'},{'Hz'}];
LinearGrid=AddOutput(LinearGrid,output);

%% Where to simulate
%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/Si2GeWaveguide'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);


%% Plot stuff

topology.field='topology';
topology.x='all';
topology.y='all';
topology.z=1;
topology.data=0;
topology.gridfield='Hz';
PlotData(topology,LinearGrid,[SimulInfo.WorkPlace 'Si2GeWaveguideOutCalFFT']);
shading interp


full=topology;
full.x='all';
full.y='all';
full.z=1;
full.field='ME2_real';
full.noplot=0;
full.data=frequency2index([SimulInfo.WorkPlace 'Si2GeWaveguideOutFFT'],c./1700e-9);
PlotData(full,LinearGrid,[SimulInfo.WorkPlace 'Si2GeWaveguideOutCalFFT']);
shading interp






%%

xint=(SiGuideZone.x:SiGuideZone.x+SiGuideZone.dx);
line=full;
line.x=Nx-20;
line.field='Ey_abs';
line.data='all';
line.noplot=1;
[datacal,pos]=PlotData(line,LinearGrid,[SimulInfo.WorkPlace 'Si2GeWaveguideOutCalFFT']);
shading interp
[datatot,pos]=PlotData(line,LinearGrid,[SimulInfo.WorkPlace 'Si2GeWaveguideOutFFT']);
datacalI=trapz(pos.x(xint),datacal(xint,:),1);
datatotI=trapz(pos.x(xint),datatot(xint,:),1);
figure
subplot(1,2,1)
plot(c./pos.y*1e6,datacalI)
subplot(1,2,2)
plot(c./pos.y*1e6,datatotI./datacalI)

%%

line.x=Nx-20;
line.field='Hz_abs';
line.data=frequency2index([SimulInfo.WorkPlace 'Si2GeWaveguideOutFFT'],c./lambda);
line.noplot=0;
[datasource,pos]=PlotData(line,LinearGrid,[SimulInfo.WorkPlace 'Si2GeWaveguideOutCalFFT']);
if LinearGrid.recomputesource
save('StephSource','datasource')
end
%%
% dataT=Extract([SimulInfo.WorkPlace 'Si2GeWaveguideOut'], 'Hz',4);
% MyLittlePlotOut(dataT)  











