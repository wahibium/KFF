%% Calculcates the frequency in frequencydomain of a certain index in out.


function freq=index2frequency(hdf5_file,index)


Datainfo.start=(hdf5read(hdf5_file,'startData'));
Datainfo.stop=(hdf5read(hdf5_file,'stopData'));
Datainfo.delta=(hdf5read(hdf5_file,'deltaData'));
frequency=(Datainfo.start:Datainfo.delta:Datainfo.stop);
freq=frequency(index);

return;