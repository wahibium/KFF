

function mycontour(myfig,pos,data,N)

[gridX,gridY]=meshgrid(pos.x,pos.y);
size(gridX)
size(gridY)
[CS,H]=contour(myfig,gridX,gridY,data',N);

end