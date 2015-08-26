%% Exports to HDF5 and copies to the destination 
% Needs:
% SimulInfo.file -> Name of the HDF5 simulation file
% SimulInfo.WorkPlace -> Directory where simulation is run
% SimulInfo.SimulDir -> Directory where simulator lives
% SimulInfo.NameSimulator -> Simulator name
% SimulInfo.PrintStep -> Steps that are printed
% SimulInfo.infile -> Input file 
function [debugout] = LocalSimulate(g,SimulInfo)
if ~exist('SimulInfo')
    SimulInfo = [];
end

%% Create workspace directory
% Default is ../Work
if ~isfield(SimulInfo, 'WorkPlace')
    SimulInfo.WorkPlace = '../Work'
end
if ~exist(SimulInfo.WorkPlace, 'dir')
    mkdir(SimulInfo.WorkPlace)
end

%% Fill default parameters if not explicitly defined
if ~isfield(SimulInfo, 'SimulatorExec')
    SimulInfo.SimulatorExec = '../../obj/fdtd'
end

if ~isfield(SimulInfo, 'PrintStep')
    SimulInfo.PrintStep = 4000;
end

if ~isfield(SimulInfo, 'infile')
    SimulInfo.infile = 'infile';
end

if ~isfield(SimulInfo, 'outfile')
    SimulInfo.outfile = './';
end

if ~isfield(SimulInfo, 'debugfile')
    SimulInfo.debugfile = 'debugout.txt';
end



%% Change cd to workdir
pushdir = pwd; % push dir
eval(['cd ' SimulInfo.WorkPlace]);
display(sprintf('pwd = %s', pwd));

%% Create an HDF5 file with simulation data
hdf_export(SimulInfo.infile,g);

%% Create directory for output files
mkdir(SimulInfo.outfile);

%% Run the simulation
runcmd = sprintf('%s -f %d -i %s -o %s -F -T', ...
    SimulInfo.SimulatorExec, ...
    SimulInfo.PrintStep, ...
    SimulInfo.infile, ...
    SimulInfo.outfile);

display(runcmd)    
system(runcmd)
debugout=0;
%eval(['cd ' pushdir]) % pop dir

