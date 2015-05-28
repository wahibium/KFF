
%% Creates a constant grid that covers the source.x,source.y,source.z
%% space in steps of source.dx,source.dy,source.dz;

%% if source.pad==1 the simulation space will be padded so that it its size
%% is divisible by 16 in the x and y direction


function [grid,source]=CreateConstantGrid(source);

N(1)=ceil(source.x/source.dx);
N(2)=ceil(source.y/source.dy);
N(3)=ceil(source.z/source.dz);
dir=[{'x'},{'y'},{'z'}];
dir2=[{'dx'},{'dy'},{'dz'}];
for cnt= (1:length(N))
    if isfield(source,'pad')

        if(source.pad==1)
            if cnt<3;
            padnum=mod(16-mod(N(cnt),16),16);
            N(cnt)=N(cnt)+padnum;
            end
        end
end
        grid.grid.(dir{cnt})=ones(1,N(cnt))*source.(dir2{cnt});
        source.(dir{cnt})=sum(grid.grid.(dir{cnt}));
end

grid=UpdataGridSizeInfo(grid);

