%% Gives the distance of the (0,0,0) point of a yee cell at index index in direction dir
%% starting from the the origin of the simulation space. 
%% Example: index2distance(LinearGrid,10,'z')

function pos=index2distance(grid,index,dir)

distance=cumsum(getfield(grid.grid,dir))-getfield(grid.grid,dir);

if index>length(distance)
        error((sprintf('The index %d exeeds the simulation space %d ',index,length(distance))));
end
pos=distance(index);



