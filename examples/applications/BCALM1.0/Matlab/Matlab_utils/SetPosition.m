function source=SetPosition(g,source) 

%% If operated in real distance units.

if isfield(source,'realdistance')
    if (source.realdistance==1)
       
            source=Real2IndexSource(g,source);
            source=rmfield(source,'realdistance');
    end   
end


%% We supply indices now. We could real positions later.

    if ischar(source.dx)
        if strcmp(source.dx,'end')
    source.dx=g.info.xx-source.x;
        end
    else
        source.dx;
    end
    
    if ischar(source.dy)
        if strcmp(source.dy,'end')
      
        source.dy=g.info.yy-source.y;
        end
    else
        source.dy;
    end
    
    if ischar(source.dz)
        if strcmp(source.dz,'end')
    source.dz=g.info.zz-source.z;
        end
    else
        source.dz;
    end  
%% Check if material is out of bounds

if (source.x+source.dx)>g.info.xx
    error('Adding material %s exceeds the gridsize in the X dimention',source.name)
end

if (source.y+source.dy)>g.info.yy
    error('Adding material %s exceeds the gridsize in the Y dimention',source.name)
end

if (source.z+source.dz)>g.info.zz
    error('Adding material %s exceeds the gridsize in the Z dimention',source.name)
end

    
%% Check source.x or source.dx is negative

if ((source.x)<0)
    error('Adding material %s has negative start in the X dimention',source.name)
end

if ((source.y)<0)
    error('Adding material %s has negative start in the Y dimention',source.name)
end

if ((source.z)<0)
    error('Adding material %s has negative start in the Z dimention',source.name)
end

if ((source.dx)<0)
    error('Adding material %s has negative d in the X dimention',source.name)
end

if ((source.dy)<0)
    error('Adding material %s has negative d in the Y dimention',source.name)
end

if ((source.dz)<0)
    error('Adding material %s has negative d in the Z dimention',source.name)
end

