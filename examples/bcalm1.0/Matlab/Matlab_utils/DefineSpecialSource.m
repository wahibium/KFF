%% This function creates different kinds of sources.
%% x,y,z,dx,dy,dz denote a box
%% omega is the frequency
%% mut is the time expetation value of the gaussian envelope
%%
%% Sourcetypes: - Constant : Source is constant over the whole selected
%%


function g=DefineSource(g,source)


%% Default name

source.name='default';


%% Initialize
source.x;
source.y;
source.z;
source.dx;
source.dy;
source.dz;
source.omega;
source.mut;
source.sigmat;
source=SetPosition(g,source); %% Set the position and verifies if out of bounds.

Nx=(source.dx+1);
Ny=(source.dy+1);
Nz=(source.dz+1);
nsource=Nx*Ny*Nz;

requiredfields=[{'x'},{'y'},{'z'},{'dx'},{'dy'},{'dz'},{'omega'},{'mut'},{'sigmat'}];

for i=1:length(requiredfields)
    if ~isfield(source,requiredfields(i))

        error(sprintf('Please specify %s',requiredfields(i)));

    end

end


field=[{'Ex'},{'Ey'},{'Ez'},{'Hx'},{'Hy'},{'Hz'}];


%% Constant Type

if strcmp(source.type,'constant')


for i=1:length(field)
    if ~isfield(source,field{i})
        source.(field{i})=zeros(1,nsource);
        else
        source.(field{i})=ones(1,nsource)*source.(field{i})(1);
        end
end
  g=AddSource(g,source);
            
end

if strcmp(source.type,'UserDefined')


for i=1:length(field)
    if ~isfield(source,field{i})
        source.(field{i})=zeros(1,nsource);
        else
        zs=(size(source.(field{i})));
        zs=[zs ones(1,3-length(zs))];
			if ((zs(1)~=Nx)||(zs(2)~=Ny)||(zs(3)~=Nz))
				display('Fieldsize:')
				zs
				display('Zonesize:')
				[Nx,Ny,Nz]
				error(['The dimentions of User Defined field ' (field{i}) 'dont match the dimentions of the zone the source is supposed to cover'] );
			end
     end
end
  g=AddSource(g,source);
            
end






%%

if strcmp(source.type,'gaussianbeam')
for i=1:length(field)
    if ~isfield(source,field{i})
        source.(field{i})=0;
    end

end
    g=AddGaussianBeam(g,source);
 
end




end













