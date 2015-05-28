function [data,pos]=mypostinterp(fem,field,x,y)


[X,Y] = meshgrid(x,y);
p = [X(:)'; Y(:)'];
res=postinterp(fem,field,p);
res = reshape(res, size(X));
figure
pcolor(X,Y,abs(res))
shading interp
pos.x=x;
pos.y=y;
data=res;



end
