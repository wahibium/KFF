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


res.startpoint=[1e16,1e15];
res.tofit=[1,1,0];
res=GetLorentzExpression(res,1);

ffit=fittype(['real(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startpoint,'Tolx',1e-10,'MaxIter',20000,'MaxfunEval',20000);
fo.robust='On';
%fo.Normalize='On';
fo.TolFun=1e-15;
%fo.lower=[1e10,0];
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,real(eps),ffit,'problem',[{0}]);

clear res;


res.tofit=[0,0,1]
res=GetLorentzExpression(res,1);
problem=[{F0.Wp1},{F0.W1}];
res.startpoint=[1e10];
ffit=fittype(['imag(',res.expr, ')'],'coeff',res.coeff,'problem',res.problem,'independent','W');
fo=fitoptions('method','nonlinearleastsquares','startpoint',res.startpoint,'Tolx',1e-10,'MaxIter',20000,'MaxfunEval',20000);
fo.TolFun=1e-15;
%fo.lower=[1e10,0];
fo.display='Iter'
ffit=setoptions(ffit,fo);
F0=fit(omega,imag(eps),ffit,'problem',problem);




%F0=fit(omega,real(eps),ffit);


Plot_Lorentz(F0,1,lambda,eps);


title('Single pole model for Gold')

xlabel('Wavelength(nm)')
ylabel('epsr')
legend('Real_Model','Imag_Model','Real_Palik','Imag_Palik')
saveas(gcf,[optpath mats{mat} 'FITSimple.eps'],'eps2c')







