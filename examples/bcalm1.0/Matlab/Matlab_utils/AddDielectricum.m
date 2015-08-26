%% Adds a dielectricum with name name at position x,y,z over a distance
%% dx,dy,dz if ambient is added as an option the dielectricum will be the
%% default material placed in first position in the dielzone array.

function g=AddDielectricum(g,name,x,y,z,dx,dy,dz)

global eps0
    source.name=name;
    source.x=x;
    source.y=y;
    source.z=z;
    source.dx=dx;
    source.dy=dy;
    source.dz=dz;

source=SetPosition(g,source); %% Computes the 'ends' checks out of bound errors 
 %% Create the field if it doesn't exist yet.
     found=0;
     
    if ~isfield(g,'deps')
        g.deps=[];
    end

    
    if ~isfield(g,'dielzone')
        g.dielzone=[];
    end
    
    
  %% Check if the dielectricum already exists.
for cnt =(1:length(g.deps))   
   
    if strcmp(g.deps(cnt).name,source.name) 
        found=1;
        g.dielzone=[g.dielzone source];
        
    end
end

if (~found) %% Put error if the dielectricum was not found.  
    error('Material %s could not be found',source.name)
end


