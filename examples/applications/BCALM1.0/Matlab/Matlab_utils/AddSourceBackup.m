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
source.dx=0;
source.dy=0;
source.dz=0;
source=SetPosition(g,source);
params(1)=source.x-1; %Indices in C are start at 0
params(2)=source.y-1;
params(3)=source.z-1;
params(4)=abs(source.Ex);
params(5)=abs(source.Ey);
params(6)=abs(source.Ez);
params(7)=abs(source.Hx);
params(8)=abs(source.Hy);
params(9)=abs(source.Hz);
params(10)=angle(source.Ex);
params(11)=angle(source.Ey);
params(12)=angle(source.Ez);
params(13)=angle(source.Hx);
params(14)=angle(source.Hy);
params(15)=angle(source.Hz);
params(16)=source.omega;
params(17)=source.mut;
params(18)=source.sigmat;


if length(params)<NPSOURCE
    warning('Source has been zero padded. Disable warnings to disable this message')
    params=[params zeros(1,NPSOURCE-length(params))];
elseif length(params)>NPSOURCE
    error('too many parameters');
end

if ~isfield(g.info,'ss')
    g.info.ss=1;
else
    g.info.ss=g.info.ss+1;
end


if ~isfield(g,'source')
    g.source(:,g.info.ss)=params;
else

     g.source(:,g.info.ss)=params;
end
