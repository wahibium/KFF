clear all
close all
clc
optpath=['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/']; % Place where I store the materials;
mats=[{'Cu'},{'Au'},{'Pt'},{'Ag'},{'Tungsten'},{'K'},{'Cr'},{'Al'}];
lstart=1000e-9;
lstop=2000e-9;
figuretop=figure;
mat=8;

% load the data

data=load([optpath mats{mat}]);
lambda=data(:,3)*1e-6;
% Just take the interesting Lambda.
lowb=find(lambda>lstart,1,'first');
highb=find(lambda>lstop,1,'first');
intb=(lowb:highb);
lambda=lambda(intb);
c=3e8;
omega=2*pi*c./lambda;
n=data(intb,4);
k=data(intb,5);
% Calculate the permittivity under the exp(i(wt-kr)) convention
eps=(n-1i*k).^2;





wpm=[1.770639638036507e+16,1.770639638036507e+16,1.332784755e16,1e16];
wm=[4.496696407505216e+14,4.496696407505216e+13,4e14,1e15];
gammam=[1.735758178726308e+12, 1.496696407505216e+12, 1e13 ,1e14];

% wpm= 1.792483390567413e+16
% wm=3.627721506779444e+14
% gammam=1.949503267892412e+14

np=2;

for pass=1
F0=1;
for p=1:np
    
    
    
    
%% Optimize for WP     
startpoint=[wm(p)];
problem=[wpm(p),0];
param=[0,1,0]
res=GetLorentzExpression(p,param,startpoint,problem,F0);
ffit=fittype(['real(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startval,'Tolx',1e-10,'MaxIter',20000,'MaxfunEval',20000);
fo.robust='On';
fo.TolFun=1e-15;
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,real(eps),ffit,'problem',res.problemval);
clear res; 

Change(2,p)=(F0.(sprintf('W%d',p))-startpoint)./startpoint;

%% Optimize for WPn     
startpoint=[wpm(p)];
problem=[F0.(sprintf('W%d',p)),0];
problem=[wm(p),0];
param=[1,0,0]
res=GetLorentzExpression(p,param,startpoint,problem,F0);
ffit=fittype(['real(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startval,'Tolx',1e-10,'MaxIter',20000,'MaxfunEval',20000);
fo.robust='On';
fo.TolFun=1e-15;
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,real(eps),ffit,'problem',res.problemval);
clear res;


Change(1,p)=(F0.(sprintf('Wp%d',p))-startpoint)./startpoint;

%% Optimize for Gp    
startpoint=[gammam(p)];
problem=[F0.(sprintf('Wp%d',p)),F0.(sprintf('W%d',p))];
param=[0,0,1]
res=GetLorentzExpression(p,param,startpoint,problem,F0);
ffit=fittype(['imag(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startval,'Tolx',1e-10,'MaxIter',20000,'MaxfunEval',20000);
fo.robust='On';
fo.TolFun=1e-15;
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,imag(eps),ffit,'problem',res.problemval);
Change(3,p)=(F0.(sprintf('G%d',p))-startpoint)./startpoint;

end


end


%% All together
% 
res=GetLorentzExpressionAll(np,0,F0);
ffit=fittype(['real(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startval,'Tolx',1e-15,'MaxIter',20000,'MaxfunEval',20000);
fo.robust='On';
fo.TolFun=1e-15;
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,real(eps),ffit,'problem',res.problemval);


res=GetLorentzExpressionAll(np,1,F0);
ffit=fittype(['imag(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startval,'Tolx',1e-15,'MaxIter',20000,'MaxfunEval',20000);
fo.robust='On';
fo.TolFun=1e-15;
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,imag(eps),ffit,'problem',res.problemval);




format long


if np==3
wpm=[F0.Wp1,F0.Wp2,F0.Wp3];
wm=[F0.W1,F0.W2,F0.W3];
gammam=[F0.G1,F0.G2,F0.G3];
end

if np==2
wpm=[F0.Wp1,F0.Wp2];
wm=[F0.W1,F0.W2];
gammam=[F0.G1,F0.G2];
end

if np==1
wpm=[F0.Wp1];
wm=[F0.W1];
gammam=[F0.G1];

end
wpm
wm
gammam

%%

Plot_Lorentz(F0,np,lambda,eps);


title('Single pole model for Gold')

xlabel('Wavelength(nm)')
ylabel('epsr')
legend('Real_Model','Imag_Model','Real_Palik','Imag_Palik')
saveas(gcf,[optpath mats{mat} 'FITSimple.eps'],'eps2c')





Change

