function g=UpdataGridSizeInfo(g)

dir=[{'x'},{'y'},{'z'}];
dir2=[{'xx'},{'yy'},{'zz'}];

for cnt=(1:length(dir))
xx=length(g.grid.(dir{cnt}));
g.info.(dir2{cnt})=xx;

end

end
