%% Gets the gaussian beam profile at distance r from the beam axis and at
%% distance z from the mimimal beamwaist w0. 
%% E=w0/(w(z))exp(-r^2/w^2(z))exp(ikz-ikr^2/(2/(R(z)))+i*xi(x))
%% w(z)=w0*sqrt(1+z/zr)
%% zr= pi*w0^2/lambda
%% R(z)=z*[1+(zr/z)^2]
%% Xi(z)=atan(z/zr)

function E=GetGaussian(r,z,omega,n,w0)
%sigmamax=3; %% if r/w>sigmamax the source is not added

z=abs(z);
c=3e8;
k=omega*n/c;
lambda=2*pi/k;
zr= pi*w0^2/lambda;
if abs(z)>0

else
R=1e20*lambda; %% Very large.
end



w=w0*sqrt(1+(z/zr)^2);

z=-z;
R=z*(1+(zr/z)^2);
xi=atan(z/zr);



%if (r/w)<sigmamax
E=w0/w*exp(-r^2/w^2)*exp(-1i*k*z-1i*k*r^2/(2*R)+1i*xi);
%else
%E=0; %% Is not added afterwards;
end