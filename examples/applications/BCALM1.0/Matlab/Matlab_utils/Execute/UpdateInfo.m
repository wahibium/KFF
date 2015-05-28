%% This function gets a grid looks at the lengths of the required fields
%% and writes them back into the grid in the info slot.



function g=UpdateInfo(g)

%% Update the simulation length

g=UpdataGridSizeInfo(g);

%% Length of the eps
if isfield(g,'deps')
g.info.ee=length(g.deps);
else
  g.info.ee=0;
end
%% Length of dielectrical zone
if isfield(g,'dielzone')
g.info.zdzd=length(g.dielzone);
else
 g.info.zdzd=0;
end
%% Length of the number of sources.
if isfield(g,'source')
% [a,b]=size(g.source);
% g.info.ss=b;
else 
g.info.ss=0;
warning('No sources in the simulation');
end

%% Number of perfectzones.
if isfield(g,'perfectlayerzone')
g.info.pp=length(g.perfectlayerzone);
else
    g.info.pp=0;
end
%% Number of lorentzzones.
if isfield(g,'lorentzzone')
g.info.zlzl=length(g.lorentzzone);
else
   g.info.zlzl=0;
end
%% Number of lorentz materials
if isfield(g,'lorentz')
g.info.ll=length(g.lorentz);
else
    g.info.ll=0;
end
%% maximum number of poles
MP=0;
for cnt= (1:g.info.ll);
    if g.lorentz(cnt).NP>MP
        MP=g.lorentz(cnt).NP;
    end
end

g.info.mp=MP; %% Memory will be assigned for MP poles for each material regardless of how many it actually has.
    
%% Number of cpmlzones.

if isfield(g,'cpmlzone')
    g.info.zczc=length(g.cpmlzone);
else
    g.info.zczc=0;

end


    
%% Number of outputzones.

if isfield(g,'outputzones')
    g.info.zozo=length(g.outputzones);
else
    g.info.zozo=0;

end


end