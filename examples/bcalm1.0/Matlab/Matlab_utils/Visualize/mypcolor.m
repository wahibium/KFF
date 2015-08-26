function mypcolor(p,x,y,z)
   x0=3*x(1)-x(2);  xe=3*x(end)-x(end-1);xm=x(1:end-1)+x(2:end);x=[x0;xm(:);xe]/2;
   y0=3*y(1)-y(2);  ye=3*y(end)-y(end-1);ym=y(1:end-1)+y(2:end);y=[y0;ym(:);ye]/2;

   
   z=[z,z(:,end);z(end,:),z(end,end)];
   z=permute(z,[2,1]);
 
   
   pcolor(x,y,z);
   %shading(p,'flat');
   %shading(p,'interp');
end
