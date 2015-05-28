%% Gives the delta of the (0,0,0) point of a yee cell in direction dir
%% starting from pos1 to pos2 of the simulation space.
%% Example: distance2delta(LinearGrid,5e-6,10e-6,'z')

function delta=distance2delta(grid,pos1,pos2,dir)

delta=distance2index(grid,pos2,dir)- distance2index(grid,pos1,dir)-1; %% Minus one because delta=0 is still a distance of one


