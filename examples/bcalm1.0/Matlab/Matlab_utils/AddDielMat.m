%% Adds a dielectricum with name name and relative perimittivity epsilon

function g=AddDielMat(g,name,epsilon,varargin)

global eps0
%% Check if the dielectricum already exists.
    source.name=name;
    source.epsilon=epsilon;
    
    
    %% Check if this is an ambient material
ambient=0;
optargin = size(varargin,2);
for cnt=(1:optargin)
    if strcmp(varargin{cnt},'ambient')
        ambient=1;
    end
end

 %% Create the field if it doesn't exist yet.
     found=0;
     
    if ~isfield(g,'deps')
        g.deps=[];
    end
    
   
for cnt =(1:length(g.deps))   
   
    if strcmp(g.deps(cnt).name,source.name) 
        found=1;
        g.deps(cnt).epsilon=source.epsilon;
        warning('Material %s already exists dielectric constant will be updated',name)
        if ambient %Switch the current one to the first one position
            tmp=g.deps(1);
            g.deps(1)=g.deps(cnt);
            g.deps(cnt)=tmp;
            warning('Material %s has allready existed but is set at ambiant',name)
        end
        
    end
end

if (~found) %% Add material
    if ambient
        g.deps=[source g.deps]; %Add as first material;
    else
        
    g.deps=[g.deps source]; %Add at the end
    end
end


