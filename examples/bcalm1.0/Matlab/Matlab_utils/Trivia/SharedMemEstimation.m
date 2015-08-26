clear all
close all 
clc
%% Plotting Graph for a Uniderectional disperion calculation. In other
%% words how many variables do we want to store in shared memory if we want
%% to update the dispertion term on the electrical field in the X
%% direction.

p=(0:8);%Number of poles
nFloats=2*(p+1); %2 per pole + the two fields

BPF=4; %number of bytes in a float
BytesperCell=BPF*nFloats; %Shared bytes needed; 
CellsperSM=[2,4,8,16].^2';%Number of cells in one thread block;
CellsperSM=[CellsperSM ;512]; % Add maximum size of threads in on thread block
TotalSM=CellsperSM*BytesperCell; % Total memory required in one thread block
MaxMem=16e3;

figure 
hold on 
color=jet(length(CellsperSM)+1);
legendarray=[];
for i=1:length(CellsperSM)

plot(p,TotalSM(i,:),'Color',color(i,:));

xlabel('Number of Poles')
ylabel('Number of Bytes in SM')
title('Cuda Memory Use for a uniaxial dispertion field update')
legendarray=[legendarray {['Cells per SM = ' num2str(CellsperSM(i))]}];
    
end
legendarray=[legendarray {'MaxMemSize'}];
plot(p,MaxMem*ones(length(p),1));

hold off

legend(legendarray)


%% Calculate How many variables we do need in Shared memory for the Curl
%% term of the electrical field update in 3 dimentions

nFloats=3+2*3 ;% 3 for the electrical field 
BytesperCell=BPF*nFloats; %Shared bytes needed; 
TotalSM=CellsperSM*BytesperCell; % Total memory required in one thread block

figure
hold on 

plot(CellsperSM,TotalSM)
plot(CellsperSM,MaxMem*ones(length(CellsperSM),1),'g');
hold off

title('Cuda Shared Memory Use for E/H curl field update in 3 dimentions')
xlabel('Cells per SM ')
ylabel('Number of Bytes in SM')
legend('Cells per SM ','MaxMemSize')




%% Everything Together

nFloats=2*3*(p)+12; %2 per pole per field direction + 6 H fields + 6 Efields
BytesperCell=BPF*nFloats; %Shared bytes needed; 
TotalSM=CellsperSM*BytesperCell; % Total memory required in one thread block
MaxMem=16e3;

figure 
hold on 
color=jet(length(CellsperSM)+1);
legendarray=[];
for i=1:length(CellsperSM)

plot(p,TotalSM(i,:),'Color',color(i,:));

xlabel('Number of Poles')
ylabel('Number of Bytes in SM')
title('Cuda Memory Use for updating everything together')
legendarray=[legendarray {['Cells per SM = ' num2str(CellsperSM(i))]}];
    
end
legendarray=[legendarray {'MaxMemSize'}];
plot(p,MaxMem*ones(length(p),1));

hold off

legend(legendarray)

%% Global memory waste by have a struct of each field containing a source
%% and all the E fields containing Lorentz poles

cubeline=(20:50:500);% number of lorentz cells in one direction 
Numberofcells=cubeline.^3;
PercentLorentz=0.2;
 %% 6 for the fields 6 four the field sources 3 for E old 6 for J,Jold per
 %% field per pole 2 for the meterial type
nFloats=6+6+(3+2*3*p)*PercentLorentz+2;
BytesperCell=BPF*nFloats; %Shared bytes needed
TotalGM=Numberofcells'*BytesperCell;

figure 
hold on 
color=jet(length(Numberofcells)+1);
legendarray=[];
for i=1:length(Numberofcells)

plot(p,TotalGM(i,:),'Color',color(i,:));

xlabel('Number of Poles')
ylabel('Number of Bytes in Global Memory')
title('Global Memory Use needed')
legendarray=[legendarray {['Cells one dimentional cubeline = ' num2str(cubeline(i))]}];
    
end
MaxMem=4e9;
legendarray=[legendarray {'MaxMemSize'}];
plot(p,MaxMem*ones(length(p),1));

hold off

legend(legendarray)

%% Hashing How many bits do we want to give to the number of
%% materials,sources,and lorentz cells

k=1;%Number of floats we use in memory
Ntot=k*32; %Number of bytes at our disposal.
Nlorentzandsources =(2.^(1:Ntot)).^(1/3);
Nmaterials=2.^(Ntot:-1:1);

figure 
semilogy((1:Ntot),Nmaterials);
hold on 
semilogy((1:Ntot),Nlorentzandsources,'g');

xlabel('Nbits in the hashing')
ylabel('Number of cells you can reach')

legend('Nmaterials','Nlorentzandsources per edge of a cube')

hold off
%% Speed estimation predictions

% Calculation for a Cell update without dispertion

CurlOp=(4+3)*6; % 4 addtions and 3 multiplications per field

DispOp=(2+2+p*(3+3))*3;% 2 Additions +2 Multiplications + p*(3addtions + 3multplications) per field

Lpercent=1; %Percentage of Lorentzcells
TotOp=1*DispOp+CurlOp;

NumberofCells=400*400*400;
TotpTimeStep=NumberofCells*TotOp;
FLOP=0.3e12;%Number of operations possible per second
TimeStepTime=TotpTimeStep/FLOP;% Time per step
nsteps=15000; %Assuming 15000 Timesteps
TotalTime=nsteps*TimeStepTime; %Estinated time of Simulation


figure

plot(p,TotalTime)
xlabel('Number of poles')
ylabel('Estimated Simul Time(s)')

title(['Cells per edge = ' num2str(NumberofCells^(1/3)) ' Flops= ' num2str(FLOP/1e9) ' Gflops Percentage Lorentz= ' num2str(Lpercent*100) '%  Number of Timesteps ' num2str(nsteps)  ])











