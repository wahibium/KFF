
%% Integrates data using pos over limit lim 
%% lim.x=6
%% lim.dx=10;
%% lim.y=1;
%% lim.dy=5;

function res=Integrate2D(pos,data,lim)



X=(lim.x:lim.x+lim.dx);
Y=(lim.y:lim.y+lim.dy);

datared=data(X,Y,:);


for cnt=1:length(pos.z)
     temp=trapz(pos.x(X),(datared(:,:,cnt)),1);
     res(cnt)=trapz(pos.y(Y),temp,2) ;
end

%% Plot an example of the fields to see if we took the zones allright.

if isfield(lim,'plotex')
    if lim.plotex~=0;
        figure
       mypcolor(gca,pos.x(X),pos.y(Y),datared(:,:,lim.plotex))
       axis equal
       shading interp
       figure
       size(data)
       size(pos.x)
       size(pos.y)
       data(:,:,lim.plotex);
       mypcolor(gca,pos.x,pos.y,data(:,:,lim.plotex)) 
        axis equal
        shading interp
    end
end

end
