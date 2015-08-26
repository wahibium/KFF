%% getGrids gets the location of a field fields in direction dir.

function out=getGrid(dir,deltaGrid,field)
 PosField=[{'Ex'},{'Ey'},{'Ez'},{'Hx'},{'Hy'},{'Hz'},{'Center'}];
 
 
  yee = [0.5, 0, 0;%Ex //distances of the fields to the centerpoint of the cell in units of gridsize(x,y,z)
        0, 0.5, 0; %Ey
        0, 0, 0.5;%Ez
        0, 0.5, 0.5;  %Hx
        0.5, 0, 0.5; %Hy
        0.5, 0.5, 0; %Hz
        0, 0, 0;] ;%Center of the cell
    

for pos=1:length(PosField)
    if strcmp(field,PosField{pos})

        out=cumsum(deltaGrid);
        out=out-(1-yee(pos,dir))*deltaGrid;
              
        
    end
end