%% Gives the index of the (0,0,0) point of a yee cell in direction dir
%% starting from the the origin of the simulation space.
%% Example: distance2index(LinearGrid,1e-6,'z')
%% If not matching, rounding to the nearest cell.
function index=distance2index(grid,pos,dir)


if (pos~=0)
    
distance=cumsum(getfield(grid.grid,dir))-getfield(grid.grid,dir);

index=find(distance<=pos,1,'last'); %% Rounded down;

if index<length(distance)
    if abs(pos-distance(index))>abs(pos-distance(index+1))
        index=index+1; %% Rounding up if t
    end
end


if abs(pos-distance(index))/pos>1e-7
    if pos-distance(end)>0
        error((sprintf('The index at position %e exeeds the simulation space %e in direction %s',pos,distance(end),dir)));
    end
    
warning(sprintf('The index at position %e was rounded of to position %e in direction %s ',pos,distance(index),dir));
end




else
    
    index=1;

end
