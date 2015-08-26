clear all
optpath=['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/']; % Place where
mats=[{'Siorig'}];    
data=load([optpath mats{1}]);

datan(:,3)=data(:,1)*1e-3;
datan(:,4)=data(:,3);
datan(:,5)=data(:,4);

fid=fopen([optpath 'Si'],'w+')

for cnt=1:length(datan(:,1))
    for cnt2=1:length(datan(1,:))
fprintf(fid,[num2str(datan(cnt,cnt2)) '\t']);
    end
fprintf(fid,'\n');
end

