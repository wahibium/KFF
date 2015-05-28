%% Plots the data of a plot object.
%% Try all the different dimentions of the output hdf5 file
%% We assume the data is always the last dimention.
%% We also assume the data Xpreceeds the Y preceeds the Z dimention.

function [data,POSITION]=PlotData(mplot,LinearGrid,hdf5_file)
global grid;
grid=LinearGrid;
%% Process limited domain

mplot=ProcessLimitedDomain(mplot);
%% Analyse what to do

fields=[{'E'},{'H'}];%% All fields
specialfields=[{'P'},{'M'}]; %% Fields that require some postprocessing

normalfield=0;
for cnt =(1:length(fields))
    if strcmp(mplot.field(1),fields{cnt})
        normalfield=cnt;
    end
end

specialfield=0;
for cnt =(1:length(specialfields))
    if strcmp(mplot.field(1),specialfields{cnt})
        specialfield=cnt;
    end
end

specialextensions=[{'phase'},{'abs'},{'complex'},{'real'},{'imag'}]; %Fields that require some postprocessing


for cnt=(1:length(specialextensions)) %% Look for extensions
    a=strfind(mplot.field,specialextensions{cnt});
    
    if ~isempty(a)
        speex(cnt)=a;
    else
        speex(cnt)=0;
    end
    
end




%% Read the the data from the file and get the axis

if sum(speex)==0
    [data,mplot,POSITION]=JustGetData(mplot,hdf5_file); %% No special extensions
end

for cnt=(1:length(specialextensions)) %% Look for extensions
    if speex(cnt)~=0
        mplot.fieldname=mplot.field;
        mplot2=mplot;
        if normalfield~=0;
            mplot2.field=[mplot.field(1:speex(cnt)-1) 'real'];
            [data,mplot2,POSITION]=JustGetData(mplot2,hdf5_file);
            mplot2.field=[mplot.field(1:speex(cnt)-1) 'imag'];
            [data2,mplot2,POSITION]=JustGetData(mplot2,hdf5_file);
            data=data+1i*data2;
        end
        
        switch specialfield
            
            case 1 % Poynting
                speex(cnt);
                mplot2.field=[mplot.field(1:speex(cnt)-2)];
                [data,mplot2,POSITION]= GetPoynting(mplot2,LinearGrid,hdf5_file);
            case 2 %% Get_Mod_E,H
                speex(cnt);
                mplot2.field=[mplot.field(1:speex(cnt)-2)];
                [data,mplot2,POSITION]= GetModE(mplot2,LinearGrid,hdf5_file);
        end
        mplot=mplot2;
        
        switch cnt
            
            case 1 %%phase
                data=angle(data);
                
            case 2 %%abs
                data=abs(data);
                
            case 3 % complex
                data=data;
            case 4 %real
                data=real(data);
            case 5%imag
                data=imag(data);
                
                
        end
        
    end
    
end

%% Check for symetryaxises
direction=[{'x'},{'y'},{'z'}];%%Direction
if isfield(mplot,'sym')
    for cnt=1:length(direction)
        if isfield(mplot.sym,(direction{cnt}))
            if abs(mplot.sym.(direction{cnt}))==1
                factor=mplot.sym.(direction{cnt});
                data=(cat(cnt,factor*flipdim(data,cnt),data));
                
                switch mplot.allcount;
                    
                    case 0 %% Single value out
                    case 1 %% 1D mplot of the data.
                        POSITION.x=[-flipud(POSITION.x); POSITION.x];
                    case 2 %% 2D case
                        if cnt==3 %% Z direction
                            POSITION.(y)=[-flipud(POSITION.(y)); POSITION.(y)];
                        else
                            POSITION.(direction{cnt})=[-flipud(POSITION.(direction{cnt})); POSITION.(direction{cnt})];
                        end
                        
                end
                size(data);
                size(POSITION.x);
                size(POSITION.y);
                
            end
        end
    end
    
    
    
    %% Plot the data using information in mplot and POSITION
    
end

if ~isfield(mplot,'noplot')
    mplot.noplot=0;
    
end
[mplot,data,POSITION]=ProcessLimitedDomainData(mplot,data,POSITION);
if mplot.noplot==0
    JustPlotData(data,mplot,POSITION);
end