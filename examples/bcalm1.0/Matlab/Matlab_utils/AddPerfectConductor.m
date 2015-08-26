%% Adds a Perfect Conductor. It basically sets the field to zero for one or serveral field components.
%% If the field component is denoted in varargin the field will be set to
%% zero. If Varargin contains 'PEC' all the the E will be set to zero. If
%% PMC all the Hx will be set to zero

function g=AddPerfectConductor(g,x,y,z,dx,dy,dz,varargin)


field=[{'Ex'},{'Ey'},{'Ez'},{'Hx'},{'Hy'},{'Hz'}];
optargin = size(varargin,2);
myfield=varargin{1};


for cnt=(1:optargin)
     

     
     if  strcmp(varargin{cnt},'PECPMC')
         myfield=field;
         optargin=6;
         
     end
 
             
end
myfield;

optargin=length(myfield);
if (optargin>6)
    error('Something went wrong here')
end

    source.name='PerfectLayer';
    source.x=x;
    source.y=y;
    source.z=z;
    source.dx=dx;
    source.dy=dy;
    source.dz=dz;
    
    
    
    source.layerindex=0;
    
    for cnt=(1:optargin)
        for cnt2=(1:length(field))
            if strcmp(myfield{cnt},field{cnt2})
                source.layerindex=source.layerindex+(2^(cnt2-1)); %% layerindex is a binary number with 6 bits. The field will be set to zero if a bit is equal to one.
            end
        end
   end

source=SetPosition(g,source); %% Computes the 'ends' checks out of bound errors 
 %% Create the field if it doesn't exist yet.
     found=0;
     
    if ~isfield(g,'perfectlayerzone')
        g.perfectlayerzone=[];
    end
 
     g.perfectlayerzone=[g.perfectlayerzone source];
        




