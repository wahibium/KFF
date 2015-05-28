function out=makemovie(mplot,LinearGrid,hdf5_file,fig)

% Get Data
mplot.noplot=1;
[data,pos]=PlotData(mplot,LinearGrid,hdf5_file); % Get the complex field.
% normalize data
data=data./max(max(abs(data)));
% Get topology

mplot.field='topology';
mplot.data=1;
[topo,pos]=PlotData(mplot,LinearGrid,hdf5_file);
maxlines=max(max(topo))
if isfield(mplot,'contour')
    maxlines=mplot.contour
end

%% Make the movie.
Nframes=20;
for phase = linspace(0,360,Nframes)
    % Phaseshift the data
    dataplot=real(data*exp(1i*phase*pi/180));
    % Make a single frame
    mypcolor(fig,pos.x,pos.y,dataplot);
    hold on
    [y,x] = meshgrid(pos.x,pos.y);
    contour(y,x,topo', maxlines,'LineColor', 'k', 'LineWidth', 1);
    caxis([-1 1]);
    hold off
    shading interp
    axis equal

    
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


out=1;
end