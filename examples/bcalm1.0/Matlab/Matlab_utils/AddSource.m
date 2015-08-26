%% This functions adds a source. We assume that 10 parameters will be
%% sufficient to hard code each source type into the kernel. By convention
%% we will use the first float to denote the type of source we use.
%% P1:x
%% P2:y
%% P3:z
%% P4:dx
%% P5:dy
%% P6:dz
%% P7:Ex
%% P8:Ey
%% P9:Ez
%% P10:Hx
%% P11:Hy
%% P12:Hz
%% P13:w
%% P14:phi
%% P15:mut of the gaussian envelope
%% P16:sigmat of the gaussian envelope 

function g=AddSource(g,source)

NPSOURCE=18; %number of parameters.

% Check if the source has really 10 paramaters if no do zero padding.

source=SetPosition(g,source);

%% Count the number of source cells.
Nx=(source.dx+1);
Ny=(source.dy+1);
Nz=(source.dz+1);
nsource=Nx*Ny*Nz;
%% Create Linear array of positions;
X=(source.x-1:source.x+source.dx-1);%Indices in C are start at 0
X=repmat(X,[1,Ny,Nz]);
X=reshape(X,[1,nsource]);


Y(1,:)=(source.y-1:source.y+source.dy-1);%Indices in C are start at 0
Y=repmat(Y,[Nx,1,Nz]);
Y=reshape(Y,[1,nsource]);

Z(1,1,:)=(source.z-1:source.z+source.dz-1)';%Indices in C are start at 0
Z=repmat(Z,[Nx,Ny,1]);
Z=reshape(Z,[1,nsource]);





params(1,:)=X; 
params(2,:)=Y;
params(3,:)=Z;
params(4,:)=abs(reshape(source.Ex,[1,nsource]));
params(5,:)=abs(reshape(source.Ey,[1,nsource]));
params(6,:)=abs(reshape(source.Ez,[1,nsource]));
params(7,:)=abs(reshape(source.Hx,[1,nsource]));
params(8,:)=abs(reshape(source.Hy,[1,nsource]));
params(9,:)=abs(reshape(source.Hz,[1,nsource]));
params(10,:)=angle(reshape(source.Ex,[1,nsource]));
params(11,:)=angle(reshape(source.Ey,[1,nsource]));
params(12,:)=angle(reshape(source.Ez,[1,nsource]));
params(13,:)=angle(reshape(source.Hx,[1,nsource]));
params(14,:)=angle(reshape(source.Hy,[1,nsource]));
params(15,:)=angle(reshape(source.Hz,[1,nsource]));
params(16,:)=repmat(source.omega,[1,nsource]);
params(17,:)=repmat(source.mut,[1,nsource]);
params(18,:)=repmat(source.sigmat,[1,nsource]);



if ~isfield(g.info,'ss')
    g.info.ss=nsource;
else
    g.info.ss=g.info.ss+nsource;
end

if ~isfield(g,'source')
    g.source(:,(1:nsource))=params;
else

     g.source(:,(g.info.ss-nsource+1:g.info.ss))=params;
end



end

