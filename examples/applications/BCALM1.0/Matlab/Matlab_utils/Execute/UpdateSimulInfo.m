function SimulInfo=UpdateSimulInfo(SimulInfo)

SimulInfo.SimulPlaceCopy=[SimulInfo.server ':' SimulInfo.SimulFileRemote];
SimulInfo.ResultWorkPlace=[SimulInfo.WorkPlace SimulInfo.outfile]; % Path to the local outputfile
SimulInfo.ResultSimulPlace=[SimulInfo.SimulPlaceCopy SimulInfo.outfile]; % Path to the remote outputfile
SimulInfo.DebugPlace=[SimulInfo.server ':' SimulInfo.DebugDir SimulInfo.DebugFile];
SimulInfo.SimulPlace=[SimulInfo.server ' ' SimulInfo.SimulDir]; % Place where the execute command is called
SimulInfo.file=[SimulInfo.WorkPlace SimulInfo.infile];% Place of the HDF5 that is created
end