clear all
close all
clc


%% GridInit



Nx=128;
Ny=128;
Nz=80;


simul.dx=30e-9;% Cell size in the X direction
simul.dy=30e-9;% Cell size in the Y direction
simul.dz=200e-9;% Cell size in the Z direction
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

lambda= 1500e-9;
omegaTM=(2*pi*3e8)/lambda;




%% Mode prediction(TE)

edr=3.14;
eclr=1.5;
e0=8.854187817e-12;
ed=edr*e0;
ecl=eclr*e0;
a=420e-9;
mu0=4*pi*1e-7;
lambda= 1400e-9;
omegaTM=(2*pi*3e8)/lambda;
u2=omegaTM^2*mu0*(ed-ecl)*a^2;




s=(0.8:0.001:0.9999)*sqrt(u2);
t=sqrt(u2-s.^2);
vector=1./s.*(besselj(1,s)./besselj(0,s))+1./t.*(besselk(1,t)./besselk(0,t)); %% TE Case
%%vector=ed./s.*(besselj(1,s)./besselj(0,s))+ecl./t.*(besselk(1,t)./besselk(0,t)); %% TM Case
plot(vector)

index1=find(vector>0,1,'last');
index2=find(vector<0,1,'first');

smode=(s(index1)+s(index2))/2;
tmode=sqrt(u2-smode^2);


hd=smode/a;
hcl=tmode/a;

gamma1=sqrt(hd^2-omegaTM^2*mu0*ed);
gamma2=sqrt(-hcl^2-omegaTM^2*mu0*ecl);
gamma2-gamma1


%% Set Matarials

%LinearGrid=AddMat(LinearGrid,'Air',1,'ambient');


Si02.name='Si02';
Si02.epsilon=eclr;
Si02.sigma=0;
Si02.ambient=1;
LinearGrid=AddMat(LinearGrid,Si02);


Si.name='Si'
Si.epsilon=edr; %% Refractive index of oxide
LinearGrid=AddMat(LinearGrid,Si); % Defining oxide
d=lambda/(2);


Sizone.name='Si',
Sizone.dx=2*a/simul.dx;
Sizone.dy=2*a/simul.dy;
Sizone.dz='end';
Sizone.x=Nx/2-Sizone.dx/2;
Sizone.y=Ny/2-Sizone.dy/2;
Sizone.z=1;
Sizone.type='CILINDER_Z'
LinearGrid=AddMatZone(LinearGrid,Sizone);
% 

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
source.dx=round(2*a/simul.dx);
source.dy=round(2*a/simul.dx);
source.x=Nx/2-source.dx/2;
source.y=Ny/2-source.dy/2;
source.dz=0;
source.z=18;
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/10;
source.Ez=1;
source.sigma=a/4;
source.type='constant';
LinearGrid=DefineSpecialSource(LinearGrid,source);


%% Define New outputs
B=2*omegaTM/(2*pi);%desired bandwidth
output.x=1;
output.y=1;
output.z=18;
output.dx='end';
output.dy='end';
output.dz=0;
output.field=[{'Ez','Hz'}];
output.deltaT=10;%floor(1/(2*B*dt));%% number of timesteps;
output.name='Optical_Fiber_Source';
output.foutstart=omegaTM*0.9/(2*pi);
output.foutstop=omegaTM*1.1/(2*pi);%% So we get all the frequencies.
output.field=[{'Ez','Hz'}];
LinearGrid=AddOutput(LinearGrid,output);
output.z=Nz-18;

output.name='Optical_Fiber_Source_end';
LinearGrid=AddOutput(LinearGrid,output);

output.name='Optical_Fiber_Source_XZ'
output.x=1;
output.z=1;
output.y=Ny/2;
output.dx='end';
output.dz='end';
output.dy=1;
LinearGrid=AddOutput(LinearGrid,output);


% output.x=1;
% output.y=Ny/2;
% output.z=1;
% output.dz='end';
% output.dx='end';
% output.dy=0;
% output.name='debug'
% LinearGrid=AddOutput(LinearGrid,output);



%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.SimulatorExec='/home/pwahl/MyCode/obj/fdtd';
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/OPITCALFIBER'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);
%% Return and Extract


plotxz.x='all'
plotxz.z='all'
plotxz.field='topology';
plotxz.y=1;
plotxz.gridfield='Ez'
plotxz.data=frequency2index([SimulInfo.outfile 'Optical_Fiber_Source_endFFT'],omegaTM/(2*pi));
  PlotData(plotxz,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_Source_XZFFT']);
  
  
  
plot0.x='all';
plot0.y='all';
plot0.z=1;
plot0.data=1;
plot0.field='topology';
plot0.gridfield='Ez';

data=PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_SourceFFT']);
PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_Source_endFFT']);



plot0.field='Ez_abs';
plot0.y='all'
plot0.data=frequency2index([SimulInfo.outfile 'Optical_Fiber_Source_endFFT'],omegaTM/(2*pi));
[data,pos]=PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_Source_endFFT']);
plot0.y=Ny/2;
plot0.data=frequency2index([SimulInfo.outfile 'Optical_Fiber_Source_endFFT'],omegaTM/(2*pi));
PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_SourceFFT']);
[dataend,positionend]=PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_Source_endFFT']);
close
plot0.field='topology';

[topoend,positionend]=PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_Source_endFFT']);
close 


plot0.x=Nx/2;
plot0.y=Ny/2;
plot0.z=1;
plot0.data='all';
plot0.field='Ez_abs';
PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_SourceFFT']);
%plot0.field='Ez';
%PlotData(plot0,LinearGrid,[SimulInfo.outfile 'Optical_Fiber_Source_end']);

%% Comparaison with theory;



for i=(1:Nx)
    
    xcalc(i)=abs(positionend.x(i)-positionend.x(Nx/2));

    if topoend(i)==4
Ez(i)=besselj(0,hd*xcalc(i))/besselj(0,hd*a);
    else
        

Ez(i)= besselk(0,hcl*xcalc(i))/besselk(0,hcl*a);
    end
end

Ez=abs(Ez);
Ez=Ez/max(Ez);

dataend=dataend/max(dataend);


figure
plot(positionend.x,Ez);








hold on
plot(positionend.x,dataend,'y*');




