function checklorentz (g,mat)

mymat=g.lorentz(mat)
eps0=8.85e-12

%calculate C1 C2 C3
mgamma=0;

%% calucate the sum of gammap

for cnt=(1:mymat.NP)
    noemer=(1+mymat.poles(cnt,3)*g.info.dt);
    mgamma=(mgamma)+(eps0*mymat.poles(cnt,1)*(mymat.poles(cnt,2)*g.info.dt)^2)/noemer;
    aplhap=(2-(mymat.poles(cnt,2)*g.info.dt)^2)/noemer
    xip=(noemer-2)/noemer
end
mgamma
noemer=2*mymat.epsilon*eps0+1/2*mgamma+mymat.sigma*g.info.dt;

C1=(1/2*mgamma)/noemer
C2=(2*mymat.epsilon*eps0-mymat.sigma*g.info.dt)/noemer
C3=2*g.info.dt/noemer