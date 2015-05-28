%% This function calculates the eps and the sigma of a material given its
%% wavelength and complex refractive index.

function Mat=GetEpsSigma(n,lambda)

c=3e8;
omega=2*pi*c/lambda;
eps=n^2;
Mat.name='NewMat';
Mat.epsilon=real(eps);
Mat.sigma=-imag(eps)*omega;



end