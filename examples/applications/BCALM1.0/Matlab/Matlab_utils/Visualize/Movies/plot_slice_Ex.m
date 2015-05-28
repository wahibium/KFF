clear all;
close all;

%% Parameters to replace axis to zero
xoffset = 4.565e-6;
yoffset = 1.921e-6;
phase = 0;

%% Load data
load Ex_cross_stubSWFWFM1.00e-07AL1.00e-07AW1.00e-07ATH8.00e-08GMW1.00e-07GW1.00e-07GD0.00e+00_Ex.mat
load TopoCrossSWFWFM1.00e-07AL1.00e-07AW1.00e-07ATH8.00e-08GMW1.00e-07GW1.00e-07GD0.00e+00_Ey.mat
pos.x = (pos.x - xoffset) * 1e6;
pos.y = (pos.y - yoffset) * 1e6;
[y,x] = meshgrid(pos.x,pos.y);

%% Start main loop
Nframes = 20;
for phase = linspace(0,360,Nframes)
    %% Plot the data
    pcolor(y,x,real(dataX' * exp(-j*phase*pi/180)));
    shading interp
    zmax = max(max(abs(dataX')));
    caxis([-1 1]*2);
    hold on;
    [y,x] = meshgrid(pos.x,pos.y);
    contour(y,x,TopoCross', [4 5], 'LineColor', 'k', 'LineWidth', 1);
    hold off;

    %% Set axis
    axis([-2 2 -1 1] * 1.2)

    %% Annotations
    annotation(gcf,'ellipse',[0.18 0.16 0.04 0.05], ...
        'LineWidth', 2);
    annotation(gcf,'ellipse',[0.195 0.18 0.01 0.01], ...
        'LineWidth', 2);
    text(-1.8, -1, 1, 'Ex', 'FontSize', 16)
    %% Process the frames
    if phase == 0
        % Save first frame here
        f = getframe;
        [im,map] = rgb2ind(f.cdata,256,'nodither');
        im(1,1,1,Nframes-1) = 0;
        k = 1;
    else
        % Save all other frames here
        f = getframe;
        im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
        k = k+1;
    end
end

imwrite(im,map,'Ex_incident.gif','DelayTime',0,'LoopCount',inf) %g443800

%% Fix axis for a printout of the file
set(gca, 'FontSize', 16, 'FontWeight', 'Bold');
xlabel('y (um)', 'FontSize', 16, 'FontWeight', 'Bold');
ylabel('x (um)', 'FontSize', 16, 'FontWeight', 'Bold');
print -dpng slice_Ex,256,'nodither');
        im(1,1,1,Nframes-1) = 0;
        k = 1;
    else
        % Save all other frames here
        f = getframe;
        im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
        k = k+1;
    end
end

imwrite(im,map,'Ex_incident.gif','DelayTime',0,'LoopCount',inf) %g443800

%% Fix axis for a printout of the file
set(gca, 'FontSize', 16, 'F