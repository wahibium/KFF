

%% Adds a CPML to the simulation. The field directions in which we have
%% attenuation are set by typing 'x+' in the varargin window. The letter
%% designates the axis along which the field is attenuated. The sign the
%% direction of the first CPML wall. For expemple x+ would mean:
%% Attenuation along the x axis and a wave propagating along the positive
%% field direction the CPML first.

%%
%% smax is the maximal electrical conductivity (give an array if you want
%% it to be anisotropic) // Eq. 7.60a in Tavlove
%% amax is the maximal frequency shift (give an array if you want it to be anisotropic)// Eq. 7.60b Tavlove
%% kmax gives the non unity real part (give an array if you want
%% anisotropy) // Eq. 7.79  Tavlove
%% m gives the exponent in the polynomials.(same for all for now)
%%

%% If border=1 The program will make a CPML that surrounds the simulation
%% space being dx dy and dz cells thick. x,y,z will be disregarded. smax
%% amax kmax will be used appropriately. Only the borders denoted by
%% direction will be added.
%% Example:
%% AddCPML(g,x,y,z,dx,dy,dz,smax,amax,kmax,m,border,'x+','y+','z+','x-','y-
%% ','z-') will sourround the simulation space completely. With cpmls dx dy
%% dz thick. Using smax kmax amax.



function g=AddCPML(g,source)
source.amax;
source.kmax;
source.m;
source.dx;
source.dy;
source.dz;


%% Calculate smax if not specified.
 if ~isfield(source,'smax')
     d=max([g.grid.x(1),g.grid.y(1),g.grid.z(1)]);
     source.smax=0.8*source.m/(d*360) %(Tavlove 7.66)
 end

%% Redundant parameters not used but may be implemented later
source.x=0;%% Possibility for adding a CPML anywhere/
source.y=0;
source.z=0;
source.border=1; % Possibility of terminating CPML with PEC or PMC
source.type='NOTERM';

source.cpmlindex=0;
direction=[{'xpos'},{'xneg'},{'ypos'},{'yneg'},{'zpos'},{'zneg'}];


for cnt=1:length(direction)

    if isfield(source,direction{cnt})
        if  source.(direction{cnt})
            source.cpmlindex=source.cpmlindex+(2^(cnt-1)); %% Creates an index acivating bits in the same order as in the direction array.
        end
    end

end

if ~isfield(g,'cpmlzone')
    g.cpmlzone=[];
end

g.cpmlzone=[g.cpmlzone source];
end