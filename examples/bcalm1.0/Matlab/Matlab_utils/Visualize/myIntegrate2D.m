
%% Integrates data using pos over limit lim 
%% lim.x=6
%% lim.dx=10;
%% lim.y=1;
%% lim.dy=5;

function res=myIntegrate2D(pos,data,lim)


if isfield(lim,'realdistance')   
    if lim.realdistance=1;
lim.z=1;%just to be able to use Real2IndexSource
lim.dz=1;
lim=Real2IndexSource(lim.g,lim) ;
    end
end


X=(lim.x:lim.x+lim.dx)
Y=(lim.y:lim.y+lim.dy)

datared=data(X,Y,:);


for cnt=1:length(pos.z)
     temp=trapz(cumsum(X),(datared(:,:,cnt)),1);
     res(cnt)=trapz(cumsum(Y),temp,2) ;
end



end
