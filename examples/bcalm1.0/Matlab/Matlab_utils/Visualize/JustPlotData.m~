%% Custom plot function
%% Plots the data: data
%% Information is in mplot
%% Axis are in POSITION





function JustPlotData(data,mplot,POSITION)




if ~isfield(mplot,'figure')
    figure
    mplot.myfig=gca;
else
    mplot.myfig=mplot.figure;
end



if isfield(mplot,'plottype')
    if strcmp(mplot.plottype,'log')
        data=10*log10(data);

    end

end





switch mplot.allcount;

    case 0 %% Single value out


    case 1 %% 1D mplot of the data.

        %%data
        %%POSITION.x
        plot(mplot.myfig,POSITION.x,data) ;
        ylabel(mplot.fieldname);
        xlabel(mplot.names{1});
        title(mplot.mtitle)


    case 2 %% 2D plot of the data.

        size(data);
        size(POSITION.x);
        size(POSITION.y);
        if isfield(mplot,'contour')&&(sum(mplot.contour)~=0)

            mycontour(mplot.myfig,POSITION,data,mplot.contour)

        else
            mypcolor(mplot.myfig,POSITION.x,POSITION.y,data);
        end
        title(mplot.mtitle);
        xlabel(mplot.names{1});
        ylabel(mplot.names{2});
        if ~ischar(mplot.data)
            axis equal
        end

    case 3 %% 3D plot of the data.
        warning('Too many dimentions to be plotted(for now, will run a python script later)')

        %         data=squeeze(data);
        %         [meshx,meshy,meshz]=meshgrid(POSITION.x,POSITION.y,POSITION.z);
        %         size(data)
        %         size(meshx)
        %
        %         contour3(mplot.myfig,meshx,meshy,meshz,permute(data,[2,1,3]));
        %         title(mplot.mtitle);
        %         xlabel(mplot.names{1});
        %         ylabel(mplot.names{2});
        %         zlabel(mplot.names{3});
    otherwise
        warning('Too many dimentions to be plotted')


end

if strcmp(mplot.field,'topology')

    hcb=colorbar('YTickLabel',mplot.cell.names);
    set(hcb,'YTickMode','manual')  ;
    set(hcb,'TickLength',[0 1]);
    set(hcb,'YTick',(mplot.cell.minvalue:1:mplot.cell.maxvalue));
    axis equal



end


if isfield(mplot,'noplot')
    if mplot.noplot==1
        close
    end
end

end