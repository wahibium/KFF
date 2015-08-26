%% Regrids a zone with a smooth transition to source.dxmin,source.dymin...
%% If source.pad==1 each direction is padded for the length to be divisible
%% by 16
%% Middle.x,y,z is the center of the encapsulated zoue in the total grid.
%% Useful to define further structures.
%% If source.downonly.(x,y,z)==1 The grid will be kept fine until the edge
%% of the grid in direction x,y,z;


function [g,middle]=encap(g,SourceArray)






dir=[{'x'},{'y'},{'z'}];
dir2=[{'dx'},{'dy'},{'dz'}];
dmin=[{'dxmin'},{'dymin'},{'dzmin'}];


for cnt=(1:length(dir))

    % Sort the SourceArray from back to begin
    for cntsource=(1:length(SourceArray))
        start(cntsource)=SourceArray{cntsource}.(dir{cnt});
        SourceArray{cntsource}.active.(dir{cnt})=1;% Activate by default
        SourceArray{cntsource}.copymiddle.(dir{cnt})=cntsource; %Copymiddle from itself,by default.
    end
    [start,index]=sort(start,'descend');

    % Check if two intersect and merge if so.

    for cntsource=(1:length(SourceArray)-1)
        bin=index(cntsource);
        binp1=index(cntsource+1);
        if SourceArray{binp1}.(dir{cnt})+SourceArray{binp1}.(dir2{cnt})>=SourceArray{bin}.(dir{cnt}); %% Overlap;
            warning(sprintf(' zone %d extends into zone %d in the %s direction: They will be merged',index(binp1),index(bin),dir{cnt}));
            SourceArray{bin}.copymiddle.(dir{cnt})=binp1;
            SourceArray{binp1}.(dir2{cnt})= SourceArray{bin}.(dir{cnt})+ SourceArray{bin}.(dir2{cnt})- SourceArray{binp1}.(dir{cnt}); %dxn=xo+dxo-xn
            SourceArray{binp1}.(dmin{cnt})=min(SourceArray{bin}.(dmin{cnt}),SourceArray{binp1}.(dmin{cnt})); %dxminn=min(dxminold,dxminnew)
            SourceArray{bin}.active.(dir{cnt})=0;%% Not activated.
            if isfield(SourceArray{bin},'downonly')%% Copy downonly if not there yet,
                if isfield(SourceArray{bin}.downonly,dir{cnt})
                    if SourceArray{bin}.downonly.(dir{cnt})==1
                        SourceArray{binp1}.downonly.(dir{cnt})=1;
                    end
                end
            end

        end


    end




    for cntsource=(1:length(SourceArray))

        source=SourceArray{index(cntsource)};
        source=SetPosition(g,source);
        
if source.active.(dir{cnt})~=0;
        if cntsource~=length(SourceArray)
            source.pad=0; % We only pad on the last shot
        end

        %% Set the downonly  to zero if they dont exist

        if ~isfield(source,'downonly')
            source.downonly.x=0;
        end

        %% Set the uponly  to zero if they dont exist

        if ~isfield(source,'uponly')
            source.uponly.x=0;
        end


        if ~isfield(source.downonly,dir{cnt})
            source.downonly.(dir{cnt})=0;
        end

        if ~isfield(source.uponly,dir{cnt})
            source.uponly.(dir{cnt})=0;
        end





        x1=source.(dir{cnt});
        x2=x1+source.(dir2{cnt});
        dx=source.(dmin{cnt});
        [newgrid,tempmiddle,insertlength]=Remesh(g,x1,x2,dx,source.downonly.(dir{cnt}),source.uponly.(dir{cnt}),dir{cnt});

        padadd=0;

        if isfield(source,'pad')
            if source.pad==1;
                if cnt<3 %% No need to pad in the z direction;
                    padnum=mod(16-mod(length(newgrid),16),16);
                    if (isfield(source,'padtype')&&strcmp(source.padtype,'symetric'))

                        if ~mod(padnum,2)
                            newgrid=[ones(1,padnum/2)*newgrid(1) newgrid ones(1,padnum/2)*newgrid(end)];
                            padadd=padnum/2;
                            tempmiddle=tempmiddle+padadd;

                        else
                            newgrid=[ones(1,(padnum+1)/2)*newgrid(1) newgrid  ones(1,(padnum-1)/2)*newgrid(end)];
                            warning('Grid Length is not really kept');
                            padadd=(padnum+1)/2;
                            tempmiddle=tempmiddle+padadd;
                        end
                    else
                        newgrid=[newgrid ones(1,padnum)*newgrid(end)];
                    end


                end

            end
        end


        %% Update the last middles
        middle(index(cntsource)).(dir{cnt})=tempmiddle; %% last added middle is the one given by the Remesh

        for (mcnt=1:cntsource-1) %% The previous middles are shifted.
            if SourceArray{index(mcnt)}.active.(dir{cnt})~=0;
            middle(index(mcnt)).(dir{cnt})= middle(index(mcnt)).(dir{cnt})+insertlength+padadd;
            end
        end


        g.grid.(dir{cnt})=newgrid;


        g=UpdataGridSizeInfo(g);
end
    end
    %% Update the merged middles
    
      for cntsource=(length(index):-1:1)
            bin=index(cntsource);
          cpyfrom=SourceArray{bin}.copymiddle.(dir{cnt});
  
          middle(bin).(dir{cnt})=middle(cpyfrom).(dir{cnt});
      end
    
    
end





end