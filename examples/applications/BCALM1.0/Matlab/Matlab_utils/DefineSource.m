%% This function creates different kinds of sources.
%% x,y,z,dx,dy,dz denote a box
%% omega is the frequency
%% mut is the time expetation value of the gaussian envelope
%%
%% Sourcetypes: - Constant : Source is constant over the whole selected
%%


function g=DefineSource(g,x,y,z,dx,dy,dz,omega,mut,sigmat,varargin)


%% Default name

source.name='default';


%% Initialize
source.x=x;
source.y=y;
source.z=z;
source.dx=dx;
source.dy=dy;
source.dz=dz;
source.omega=omega;
source.mut=mut;
source.sigmat=sigmat;

field=[{'Ex'},{'Ey'},{'Ez'},{'Hx'},{'Hy'},{'Hz'}];

source=SetPosition(g,source); %% Set the position and verifies if out of bounds.
Amp=zeros(1,length(field));

%% Get the fields
optargin = size(varargin,2);
for cnt=(1:optargin)
    for cnt2=(1:length(field))
        if strcmp(varargin{cnt},field{cnt2})
            Amp(cnt2)=(varargin{cnt+1});
        end
    end
end

%% Look for type
for cnt=(1:optargin)
    if strcmp(varargin{cnt},'type')
        type=varargin{cnt+1};
    end
end


%% Constant Type

if strcmp(type,'constant')

    for cnt=(1:length(field))
        source=setfield(source,field{cnt},Amp(cnt)); % set constant field everywhere.

    end

    xstarttemp=source.x;
    ystarttemp=source.y;
    zstarttemp=source.z;
    
    for x=(xstarttemp:xstarttemp+source.dx)
        for y=(ystarttemp:ystarttemp+source.dy)
            for z=(zstarttemp:zstarttemp+source.dz)

                    source.x=x;
                    source.y=y;
                    source.z=z;
          
                g=AddSource(g,source);




            end
        end
    end

end


