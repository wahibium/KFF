%% Exports to HDF5 and copies to the destination 

function [debugout] = ExportSimulateReturn(g,SimulInfo)


%% Create an HDF5 file
if strcmp(SimulInfo.WhichCode,'Mine')
hdf_export(SimulInfo.file,g);
end

%% Copies the Required HDF5 files to the destination
CpyFiles({SimulInfo.file},SimulInfo.SimulPlaceCopy,SimulInfo);




if isfield(SimulInfo,'copyonly')
    if SimulInfo.copyonly==1
        debugout=0;
        return;
    end
end
%% Run the simulation
RunSimul(SimulInfo.SimulPlace,SimulInfo.Port,SimulInfo.NameSimulator,['-f ' num2str(SimulInfo.PrintStep) ' -i '  '''' [SimulInfo.SimulFileRemote SimulInfo.infile] ''''  ' -o ' '''' [SimulInfo.SimulFileRemote] '''' ' -F -T' ]);


%% Copy debug files back

system(['scp ' SimulInfo.DebugPlace ' ' SimulInfo.WorkPlace]);

%% Get the debugfiles

debugout=load([SimulInfo.WorkPlace  SimulInfo.DebugFile]);
%debugout=0;


end