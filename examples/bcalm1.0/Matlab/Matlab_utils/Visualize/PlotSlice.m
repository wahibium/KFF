%% Plots a slice of X, uses the dimentions in cells provided in X


function PlotSlice(p,X,N,M)

X=reshape(X,N,M);

[x,y]=meshgrid((1:N),(1:M));
x=(1:N);
y=(1:M);
mypcolor(get(p,'Children'),x,y,X);
axis equal % equal tick marks and no stretch-to-fill
% shading flat
%shading faceted
shading interp 
   

    
end

   