close all
clear all
clc



Nx=64;
Ny=64;
Nz=64;%% 3 layers to do a 2D simulation 


simul.dz=50e-9;% Cell size in the X direction
simul.dx=5e-9;% Cell size in the Y direction
simul.dy=25e-9;% Cell size in the Z direction
simul.pad=1;% Pad in the X and Y direction so that the size is divisible by 16

simul.x=Nx*simul.dx;%% Approximate total simulation size in the X direction
simul.y=Ny*simul.dy;%% Approximate total simulation size in the Y direction
simul.z=Nz*simul.dz;%% Approximate total simulation size in the Z direction
[LinearGrid,simul]=CreateConstantGrid(simul); %% Create a constant Grid

c=3e8;% Speed of light
dt=0.9*1/(c*sqrt(1/(simul.dx^2)+(1/simul.dy^2)+(1/simul.dz)^2)); % Timestep using a courant factor of 0.9
nsteps=1000;% Number of timesteps
LinearGrid.info.dt=dt; % Set the timestep
LinearGrid.info.tt=nsteps; %Set the number of timesteps

lambda= 1500e-9;
omegaTM=(2*pi*3e8)/lambda;



%% Set Matarials
wpm=[1.014076882e16,6.861574653e15,1.832181311e15];
wm=[0,2.046174861e10,8.225155873e14];
gammam=0*[0,4.390839341e14,3.52118567e15];

wpm=[1.014076882e16];
wm=[0];
gammam=0*[0];



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
plot(lambdatest,imag(epsr),'r*');
Au.poles=[wpm',wm',gammam'];
Au.name='Au';
Au.epsilon=1;
LinearGrid=AddMat(LinearGrid,Au);

Air.name='Air';
Air.epsilon=1;
Air.sigma=0;
Air.ambient=1;
LinearGrid=AddMat(LinearGrid,Air);

%% Add Materials
d=50e-9;
Auzone.name='Au';
Auzone.x=1;
Auzone.dx=Nx/2-floor(d/simul.dx);
Auzone.z=1;
Auzone.dz='end';
Auzone.y=1;
Auzone.dy='end';
LinearGrid=AddMatZone2(LinearGrid,Auzone);
Auzone.x=Nx-Auzone.dx;
LinearGrid=AddMatZone2(LinearGrid,Auzone);




%% Add CPML
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
source.dz=0;
source.dx=0;
source.dy='end';
source.z=25;
source.x=Nx/2;
source.y=1;
source.omega=omegaTM;
source.mut=nsteps/4;
source.sigmat=nsteps/10;
%%source.Ez=1; %TM
source.Hy=1;%TE 

source.type='constant';
LinearGrid=DefineSpecialSource(LinearGrid,source);
%% Add Perfect Protector
%LinearGrid=AddPerfectConductor(LinearGrid,1,source.y+5,source.z,source.x,5,'end',{'Hz'});


B=2.5*omegaTM/(2*pi); %desired bandwidth
output.z=1;
output.x=1;
output.y=Ny/2;
output.dz='end';
output.dx='end';
output.dy=0;
output.deltaT=floor(1/(2*B*dt));%% number of timesteps;
output.name='inclined_slab';
output.average=1;
output.foutstart=omegaTM*0.7/(2*pi);
output.foutstop=omegaTM*1.3/(2*pi);%% So we get all the frequencies.
output.field=[{'Hy'}];
LinearGrid=AddOutput(LinearGrid,output);

%% Section 7: Simulate!
SimulInfo.WorkPlace = '/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %% Where to work
SimulInfo.SimulatorExec='/home/pwahl/CUDA_SIMULATIONS/workspace/obj/fdtd'; %% Place where the fdtd program is located
SimulInfo.SimulatorExec='/home/pwahl/MyCode/obj/fdtd';
SimulInfo.infile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/MIMISO'; %% Name of the input HDF5 file
SimulInfo.outfile='/home/pwahl/CUDA_SIMULATIONS/TestCaseResults/'; %Place where to output the HDF5 file
[debugout] = LocalSimulate(LinearGrid, SimulInfo);

%% Copy Back


% system(['scp tullibardine:' SimulInfoM.SimulFileRemote 'inclined_slab' ' ' SimulInfo.outfile]);
%  dataT=Extract([SimulInfo.outfile 'inclined_slab'], 'Hz',4);
% 
%  MyLittlePlotOut(dataT)  

topology.field='topology';
topology.z='all';
topology.x='all';
topology.y=1;
topology.data=0;
topology.gridfield='Hy';
topology.gridfield='Ex'
[topologydata,posdata]=PlotData(topology,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);
shading interp


full=topology;
full.z='all';
full.x='all';
full.y=1;
full.plottype='normal';
full.field='Hy_real';
full.noplot=0;
full.data=frequency2index([SimulInfo.outfile 'inclined_slabFFT'],omegaTM/(2*pi));
PlotData(full,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);
shading interp

line=full;
line.x=Nx/2;
PlotData(line,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);




test=topology;
test.field='Hy_abs';
test.z=2*Nz/4;
test.x='all';
test.y=1;
test.plottype='normal';
test.noplot=0;
test.data=frequency2index([SimulInfo.outfile 'inclined_slabFFT'],omegaTM/(2*pi));
[testdata,pos]=PlotData(test,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);


sourcecheck=test;
sourcecheck.x=source.x;
sourcecheck.y=source.y
sourcecheck.z=source.z;
sourcecheck.dx=0;
sourcecheck.dy=0;
sourcecheck.dz=0;
sourcecheck.data='all';
PlotData(sourcecheck,LinearGrid,[SimulInfo.outfile 'inclined_slabFFT']);


%% Theory for the TM0 mode


% Get the permittivity

omegath=omegaTM;
k0th=omegath/c;
index=find(omegatest>omegath,1,'last');
epsmth=epsr(index)*eps0;
epsith=1*eps0;

% Dimentions of the guide
a=d;

% Solving the dispertion relation numerically.
fh=@(ki) tanh(ki*a)^2-(ki^2+omegath^2*mu0*(epsith-epsmth))*epsith^2/(epsmth^2*ki^2);
[kis]=fzero(fh,k0th);

kms=sqrt(kis^2+(omegath^2*mu0*(epsith-epsmth)));
kzs=sqrt(kis^2+omegath^2*mu0*epsith);
lzs=2*pi/kzs;


fhhyp=@(ki,km) km^2-ki^2-omegath^2*mu0*(epsith-epsmth);
fhtanh=@(ki,km) tanh(ki*a)+(km/epsmth)/(ki/epsith);


%% Get a Y axis centered in the middle of the guide;

yaxis=pos.x-pos.x(Ny/2);

%% Filling in the Values.
for j=(1:Ny)
    
    y=yaxis(j);
    
    if abs(y)>a
        psi(j)=exp(-kms*(abs(y)-a));
    else
        psi(j)=cosh(kis*y)/cosh(kis*a);
    end

end

figure
hold on
plot(pos.x,testdata/max(testdata))
plot(pos.x,psi,'y*')
hold off













