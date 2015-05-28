clear all
close all
clc
optpath=['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/']; % Place where I store the materials;
mats=[{'Cu'},{'Au'},{'Pt'},{'Ag'},{'Tungsten'},{'K'},{'Cr'},{'Al'}];
lstart=650e-9;
lstop=1200e-9;
figuretop=figure;
mat=2;

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


wpm=[4.386227824e15,1.008519648e15,7.251716026e15,1.332784755e16];
wm=[0,1.874386332e15,3.534954035e15,0];
gammam=[1.565488534e15 8.109675964e14 0 ,0];


% wpm=[4.386227824e15,1.008519648e15,7.251716026e15,1.332784755e16];
% wm=[1e15,1.874386332e16,3.534954035e16,1e15];
% gammam=[1.565488534e15 8.109675964e14 1e10 ,1e10];
% 
% wpm= 1.142201622843792e+16;
% wm=8.346384313840495e+14;
% gammam= 9.907739353915755e+13;

np=3;

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
%% Optimize for WPn     
startpoint=[wpm(p)];
problem=[F0.(sprintf('W%d',p)),0];
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


end


end

%% All together

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







