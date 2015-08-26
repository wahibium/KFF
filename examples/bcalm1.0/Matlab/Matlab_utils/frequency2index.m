%% Calculcates the index in frequencydomain of a certain frequency in out.


function index=frequency2index(hdf5_file,freq)


Datainfo.start=(hdf5read(hdf5_file,'startData'));
Datainfo.stop=(hdf5read(hdf5_file,'stopData'));
Datainfo.delta=(hdf5read(hdf5_file,'deltaData'));
frequency=(Datainfo.start:Datainfo.delta:Datainfo.stop);
index=find(frequency<=freq,1,'last');

if (freq<Datainfo.start||freq>Datainfo.stop)
    error('The frequency%e is not present in the outputfile fstart=%e,fstop=%e',freq,Datainfo.start,Datainfo.stop)
    
end


if frequency(end)<freq
    error('The asked frequency is larger than the Nyquist frequency');
end



if abs(freq-frequency(index))/freq>1e13
    
warning(['MS_ROUND_FREQ',sprintf('The frequency %e was rounded of to %e ',freq,frequency(index))]);
end