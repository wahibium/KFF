%% This script should extract and process the optical constants of
%% differant metals from Palik and generate plots of them. It should also
%% calculate lorentz pole models for the different metals.

clear all
close all
clc
optpath=['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/']; % Place where I store the materials;

mats=[{'Cu'},{'Au'},{'Pt'},{'Ag'},{'Tungsten'},{'K'},{'Cr'},{'Al'},{'Si'}];

lstart=200e-9;
lstop=1000e-9;
figuretop=figure;
for mat=1:length(mats)

    % load the data
    
    data=load([optpath mats{mat}]);
    lambda=data(:,3)*1e-6;
    % Just take the interesting Lambda.
    lowb=find(lambda>lstart,1,'first');
    highb=find(lambda>lstop,1,'first');
    intb=(lowb:highb);
    lambda=lambda(intb);
    LAMBDA{mat}=lambda;
    n=data(intb,4);
    k=data(intb,5);
    % Calculate the permittivity under the exp(i(wt-kr)) convention
    eps=(n-1i*k).^2;
    
    
    
    [AX,H1,H2]=plotyy(lambda*1e9,real(eps),lambda*1e9,imag(eps));
    H=[H1,H2];
   
    title(['Optical constants of ' mats{mat}])

    for cnt=(1:length(AX))
       set(AX(cnt),'ycolor','k')
       set(AX(cnt),'fontsize',14)
       set(H(cnt),'LineWidth',2)
       set(AX(cnt),'LineWidth',2)
       set(get(AX(cnt),'yLabel'),'FontSize',14)
       set(get(AX(cnt),'xLabel'),'FontSize',14)
       set(get(AX(cnt),'title'),'FontSize',14)
    end
    
    set(get(AX(1),'yLabel'),'string','{\epsilon_{real}}')
    set(get(AX(2),'yLabel'),'string','{\epsilon_{imag}}')
    set(get(AX(2),'xLabel'),'string','{\lambda} (nm)')
    legend('Imag','Real')
    
    saveas(figuretop,['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/eps_' mats{mat}],'eps2c')
     
   ratio{mat}=abs(imag(sqrt(eps./(1+eps))));
    plot(lambda*1e9,ratio{mat},'LineWidth',2)
    title(['Ratio of  ' mats{mat}])
    set(get(gca,'title'),'FontSize',14);
    xlabel('{\lambda} (nm)')
    ylabel('sqrt(imag({\epsilon}/(1+{\epsilon}))')
    set(get(gca,'yLabel'),'FontSize',14);
    set(get(gca,'xlabel'),'FontSize',14);
    set(gca,'fontsize',14);

    saveas(figuretop,['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/ratio_' mats{mat}],'eps2c')
    
    
end
%%
figure
mcol=hsv(length(mats))
hold on
set(gca,'yscale','log')
for mat=1:length(mats)
   
    semilogy(LAMBDA{mat},ratio{mat},'color',mcol(mat,:),'Linewidth',2)
    
    
end

legend(mats)
ylabel('sqrt(imag({\epsilon}/(1+{\epsilon}))')
 title(['Ratio of All Materials'])
    saveas(gcf,['/home/pwahl/CUDA_SIMULATIONS/workspace/Matlab/Matlab_utils/Materials/All'],'eps2c')
