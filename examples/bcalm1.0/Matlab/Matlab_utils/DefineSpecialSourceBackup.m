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



requiredfields=[{'x'},{'y'},{'z'},{'dx'},{'dy'},{'dz'},{'omega'},{'mut'},{'sigmat'}];

for i=1:length(requiredfields)
    if ~isfield(source,requiredfields(i))

        error(sprintf('Please specify %s',requiredfields(i)));

    end

end


field=[{'Ex'},{'Ey'},{'Ez'},{'Hx'},{'Hy'},{'Hz'}];

source=SetPosition(g,source); %% Set the position and verifies if out of bounds.


for i=1:length(field)
    if ~isfield(source,field{i})

        source.(field{i})=0;
    else

        nonzero=field{i};%% Field that needs to be updated.
        nonzerovalue=source.(field{i});

    end

end






%% Constant Type

if strcmp(source.type,'constant')


    start(1)=source.x;
    start(2)=source.y;
    start(3)=source.z;
    stop(1)=source.x+source.dx;
    stop(2)=source.y+source.dy;
    stop(3)=source.z+source.dz;
   g=preallocatesource(g,start,stop)   ;
          
    for x=(start(1):stop(1))
        for y=(start(2):stop(2))
            for z=(start(3):stop(3))

                source.x=x;
                source.y=y;
                source.z=z;
                source.dx=0;
                source.dy=0;
                source.dz=0;
                g=AddSource(g,source);
                


            end
        end
    end

end


%% Gaussian

if strcmp(source.type,'gaussian')
    if ~isfield(source,'sigma')
        error('Please specify the sigma')
    end



    xstarttemp=source.x;
    ystarttemp=source.y;
    zstarttemp=source.z;

    xmiddle=(index2distance(g,xstarttemp,'x')+index2distance(g,xstarttemp+source.dx,'x'))/2;
    ymiddle=(index2distance(g,ystarttemp,'y')+index2distance(g,ystarttemp+source.dy,'y'))/2;
    zmiddle=(index2distance(g,zstarttemp,'z')+index2distance(g,zstarttemp+source.dz,'z'))/2;

    for x=(xstarttemp:xstarttemp+source.dx)
        xgauss=exp(-(index2distance(g,x,'x')-xmiddle)^2/source.sigma^2);

        for y=(ystarttemp:ystarttemp+source.dy)
            ygauss=exp(-(index2distance(g,y,'y')-ymiddle)^2/source.sigma^2);

            for z=(zstarttemp:zstarttemp+source.dz)
                zgauss=exp(-(index2distance(g,z,'z')-zmiddle)^2/source.sigma^2);


                source.x=x;
                source.y=y;
                source.z=z;


                source.(nonzero)=nonzerovalue*xgauss*ygauss*zgauss;
                out(x-xstarttemp+1,y-ystarttemp+1,z-zstarttemp+1)= source.(nonzero);
                g=AddSource(g,source);

            end
        end
    end




end



%%

if strcmp(source.type,'gaussianbeam')

    g=AddGaussianBeam(g,source);
    
end



%% sinus


%source(x,y,z)=sin(nx*pi*x/dx)*sin(ny*pi*y/dy)*sin(nz*pi*z/dz)
%if dx,dy,dz = zero the sinus term in the product will be set exqual to one.

if strcmp(source.type,'sinus')



    xstarttemp=source.x;
    ystarttemp=source.y;
    zstarttemp=source.z;

    dx=index2distance(g,xstarttemp+source.dx,'x')-index2distance(g,xstarttemp,'x');
    dy=index2distance(g,ystarttemp+source.dy,'y')-index2distance(g,ystarttemp,'y');
    dz=index2distance(g,zstarttemp+source.dz,'z')-index2distance(g,zstarttemp,'z');

    for x=(xstarttemp:xstarttemp+source.dx)
        if (dx>0)
            xsin=sin(source.nx*pi*(index2distance(g,x,'x')-dx)/dx);
        else
            xsin=1;
        end
        for y=(ystarttemp:ystarttemp+source.dy)
            if (dy>0)
                ysin=sin(source.ny*pi*(index2distance(g,y,'y')-dy)/dy);
            else
                ysin=1;
            end
            for z=(zstarttemp:zstarttemp+source.dz)
                if (dz>0)
                    zsin=sin(source.nz*pi*(index2distance(g,z,'z')-dz)/dz);
                else
                    zsin=1;
                end


                source.x=x;
                source.y=y;
                source.z=z;


                source.(nonzero)=nonzerovalue*xsin*ysin*zsin;
                out(x-xstarttemp+1,y-ystarttemp+1,z-zstarttemp+1)= source.(nonzero);
                g=AddSource(g,source);

            end
        end
    end




end


end













