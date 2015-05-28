%% Processess the limited domains


function [mplot,data,POSITION]=ProcessLimitedDomainData(mplot,data,POSITION)

directions=[{'x'},{'y'},{'z'}];

%% take care of the symmetry
for (cnt=1:mplot.allcount)
    if POSITION.(directions{cnt})(1)<0 %% Then we have symmety
        if isfield(mplot,'limitfield')
            if isfield(mplot.limitfield,directions{cnt})
                mend=mplot.limitfield.(directions{cnt})(end); % end of the axis
                l=length(POSITION.(directions{cnt})); % size of 2 Nx
                mplot.limitfield.(directions{cnt})= (l/2-mend+1:l/2+mend+1);
            end
            
        end
    end
end




switch mplot.allcount;
    
    
    case 0
        %do nothting
    case 1
        if isfield(mplot.limitfield,'x')
            data=data(mplot.limitfield.x);
            POSITION.x=POSITION.x(mplot.limitfield.x);
        end
    case 2
       
            if isfield(mplot.limitfield,'x')
            data=data(mplot.limitfield.x,:);
            POSITION.x=POSITION.x(mplot.limitfield.x);
            end
        
            if isfield(mplot.limitfield,'y')
            data=data(:,mplot.limitfield.y);
            POSITION.y=POSITION.y(mplot.limitfield.y);
            end
    case 3
        
            if isfield(mplot.limitfield,'x')
            data=data(mplot.limitfield.x,:,:);
            POSITION.x=POSITION.x(mplot.limitfield.x);
            end
        
            if isfield(mplot.limitfield,'y')
            data=data(:,mplot.limitfield.y,:);
            POSITION.y=POSITION.y(mplot.limitfield.y);
            end        

            if isfield(mplot.limitfield,'z')
            data=data(:,:,mplot.limitfield.z);
            POSITION.z=POSITION.z(mplot.limitfield.z);
            end        

end
