%% Sets the amount of bits used for the storage of each variable.

function g=SetBits(g,type)

if strcmp(type,'regular')
    
    
    g.bit.deps=8; % Amount of for the amount the storage address of the dielectric properties in the constant memory
    g.bit.source=20; % Amount of bits used for the sources
    g.bit.perfectlayer=6; % Amount of bits used for the layers
    
end