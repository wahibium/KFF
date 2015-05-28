
function [A] = Extract (filename, dataname,dim)


A = double (permute (hdf5read (filename, dataname), (1:dim)));
dims = size (A);

reshapehelp=[];
for i=(1:dim)
    reshapehelp=[dims(i) reshapehelp];
end

	A = reshape (A, reshapehelp);
    
 
