%% Return epsilon of metal from Lorentzian poles

function epsr=GetLorentzEpsr(poles,omega)

wpm = poles(:,1);
wm = poles(:,2);
gammam = poles(:,3);

epsr=1;
eps0=8.854187817E-12;
mu0=4*pi*1e-7;
for m=(1:length(wpm))    
   epsr=epsr+ (wpm(m))^2./(wm(m)^2-omega.^2+i*omega*gammam(m));
%    alpha(m) = (2-wm(m)^2*dt^2)/(1+dt*gammam(m)/2);
%    zeta(m)  = (dt*gammam(m)/2 - 1) / (dt*gammam(m)/2 + 1);
%    gamma(m) = (wpm(m)*wpm(m)*eps0*dt*dt) / (dt*gammam(m)/2 + 1);
%    sumgamma=gamma(m)+sumgamma;     
end

