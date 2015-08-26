
topology.x='all'
topology.y='all'
topology.z=1
topology.field='Ez'

hdf5_file=[SimulInfoM.WorkPlace 'testout'];
mattype=Extract(hdf5_file,'mattype',3);
property=Extract(hdf5_file,'property',3);

dim=size(property);






