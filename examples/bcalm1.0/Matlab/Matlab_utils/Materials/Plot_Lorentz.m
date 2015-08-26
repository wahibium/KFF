function Plot_Lorentz(FO,np,lambda,EPSR);

c=3e8;
lambda;
lf=(lambda(1):(lambda(end)-lambda(1))/200:lambda(end));
omega=2*pi*c./lf;
omega2=2*pi*c./lambda;

index=1;

epsr=ones(1,length(lf));
epsr2=ones(1,length(lambda))';
for cnt =(1:np)
wp=FO.(sprintf('Wp%d',cnt));
w=FO.(sprintf('W%d',cnt));
G=FO.(sprintf('G%d',cnt));
epsr=epsr+((wp)^2)./(w^2-omega.^2+1i.*omega*G);
epsr2=epsr2+((wp)^2)./(w^2-omega2.^2+1i.*omega2*G);

    
    

end

   
figure    
plot(lf*1e9,real(epsr),lf*1e9,imag(epsr));
hold on
plot(lambda*1e9,real(EPSR),'o',lambda*1e9,imag(EPSR),'o');
figure
plot(lambda*1e9,real(EPSR-epsr2)./real(epsr2)*100,'o');
figure
plot(lambda*1e9,imag(EPSR-epsr2)./imag(epsr2)*100,'o')

end

