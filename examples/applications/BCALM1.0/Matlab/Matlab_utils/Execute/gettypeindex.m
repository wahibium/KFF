function type=gettypeindex(typename)

 BOX        =0;  % fills the whole volume fills a box between x y z and x+dx y+dy z+dz
 CILINDER_X =1;  % circle in yz cilinder in x fills a box between x y z and x+dx y+dy z+dz
 CILINDER_Y =2;  % circle in xz cilinder in y fills a box between x y z and x+dx y+dy z+dz
 CILINDER_Z =3;  % circle in xy cilinder in z fills a box between x y z and x+dx y+dy z+dz
 BALL       =4;   % spehrical in the 3 directions fills a box between x y z and x+dx y+dy z+dz

 
 if strcmpi(typename,'BOX')
     type=BOX;
 elseif strcmpi(typename,'CILINDER_X')
     type=CILINDER_X;
 elseif strcmpi(typename,'CILINDER_Y')
     type=CILINDER_Y;
 elseif strcmpi(typename,'CILINDER_Z')
     type=CILINDER_Z;
 elseif strcmpi(typename,'BALL')
     type=BALL;
 else
     error('Invalid type %s',typename);
 end
 
 
end

