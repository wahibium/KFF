%% Adds a dielectricum or a lorentz material with name name at position x,y,z over a distance
%% dx,dy,dz 



function g=AddMatZone(g,source)


if isfield(source,'calibrate')&& (source.calibrate==1)
    if isfield(g,'calibrate')&& (g.calibrate==1)
        return;
    end
end


requirednames=[{'name'},{'x'},{'y'},{'z'},{'dx'},{'dy'},{'dz'}];
A=(isfield(source,requirednames));
[temp,index]=min(A);
if (temp==0)   
error('Please set %s',requirednames{index});  
end

if ~(isfield(source,'type'))
    source.type='box';
end

source=SetPosition(g,source); %% Computes the 'ends' checks out of bound errors 
 %% Create the field if it doesn't exist yet.
     found=0;
     
    if ~isfield(g,'deps')
        g.deps=[];
    end

    
    if ~isfield(g,'dielzone')
        g.dielzone=[];
    end
    
    if ~isfield(g,'lorentzzone')
        g.lorentzzone=[];
    end 
    
    
 
  %% Check if the dielectricum already exists.
for cnt =(1:length(g.deps))   
   
    if strcmp(g.deps(cnt).name,source.name) 
        found=1;
        g.dielzone=[g.dielzone source];
        
    end
end

   if isfield(g,'lorentz')
  
    
for cnt =(1:length(g.lorentz))   
   
    if strcmp(g.lorentz(cnt).name,source.name) 
        if (found==1)
            error('Matertial %s seems to be defined both as a lorentz material and a usual material',source.name)
        end
        found=1;
        g.lorentzzone=[g.lorentzzone source];
        
    end
end
   end

if (~found) %% Put error if the dielectricum was not found.  
    error('Material %s could not be found',source.name)
end


