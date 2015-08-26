%% Changes a dimention limit in the a direction to all and keeps track of
%% the indices so that at the end of the road the right plot can be
%% generated.

function mplot=ProcessLimitedDomain(mplot)

directions=[{'x'},{'y'},{'z'}];

limitnum=1;
mplot.limitfield=[];

for cnt=1:length(directions)

    if isnumeric(mplot.(directions{cnt}))
        if length(mplot.(directions{cnt}))>1
                     
            
            mplot.limitfield.(directions{limitnum})=mplot.(directions{cnt}); % save the indexes
            mplot.(directions{cnt})='all'; %put the particular direction to all
            limitnum=limitnum+1;
       end
    end   
end



end