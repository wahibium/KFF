%% Tranforms a whole source from real distances to indexes


function source=Real2IndexSource(g,source)

       dir=[{'x'},{'y'},{'z'}];
       dir2=[{'dx'},{'dy'},{'dz'}];
       for cnt=(1:length(dir))
           tempdir=getfield(source,dir{cnt}); %get x,y,z
           source=setfield(source,dir{cnt},distance2index(g,tempdir,dir{cnt})) ;  %set x,y,z in index units
           tempdir2=getfield(source,dir2{cnt});
           if ~isstr(tempdir2)
        
           source=setfield(source,dir2{cnt},distance2delta(g,tempdir,tempdir+tempdir2,dir{cnt}));   %set x,y,z in index units

           end
       end 
       
       source.realdistance=1;


end