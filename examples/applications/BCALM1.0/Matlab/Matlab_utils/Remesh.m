function [dddx,middle,insertlength] = Remesh(g, index1,index2,dx2,downonly,uponly,dir)


dddx=g.grid.(dir);
dx1=dddx(index1);
factor=0.5;
NUMLAYERS=ceil(log(dx2/dx1)/log(factor));

initlength=length(dddx);%% Save the intitial length


if uponly==0
    first=NUMLAYERS+1+index1;
    added=NUMLAYERS;
else
    first=index1;
    added=0;
end


%% First decent

if uponly==0
    for itr = 1:NUMLAYERS
        insert=[dddx(index1)*factor dddx(index1)*factor];
        dddx = [dddx(1:index1-1) insert  dddx(index1+1:end)];
        index1 = index1+1;
    end
end
index2original=index2;
index2=index2+added;

if downonly==0
    added=added+NUMLAYERS;
    last=index2original+added-NUMLAYERS-1;
else
    last=index2original+added;
end



if downonly==0
    %% Back Up
    for itr = 1:NUMLAYERS
        insert=[dddx(index2)*factor dddx(index2)*factor];
        dddx = [dddx(1:index2-1) insert  dddx(index2+1:end)];
    end
end
%% Middle

insert=ones(1,(last-first+1)*factor^(-NUMLAYERS))*dx1*factor^NUMLAYERS;


dddx=[dddx(1:first-1) insert dddx(last+1:end)];

insertlength=length(dddx)-initlength; %% number of elements added;

middle=ceil(length(insert)/2+first-1);%% For getting the middle




