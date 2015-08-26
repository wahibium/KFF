%% Sets in grid g the field 'field' designated by Bornum on a Thickeness 'Thick' to value 'Value'
% Bornum binary number z2z1-y2y1-x2x1 given as a string.
% EX: If x2 is 1 then the second border normal to the x direction will be updated.
%% If type = 'perfectlayerzone'
% then the following strings in  varargin contain the parameters to use the function AddPerfectConductor
%% If type=cpml My_SetAllBorders will create CPML along all borders of
% Thick cells thick.
%The x y z normal boundaries:   will be only be aborsive in
%   the X Y Z directions.
%Edges:  will be absorbant in all directions execpt
%        the direction the edge is oriented to.
%Corners: will be absorbant in all directions.

function g=SetAllBorders(g,Bornum,Thick,type,varargin)

optargin = size(varargin,2);
varargin;

if ~strcmp(type,'cpml')
    %% Update first border on X direction
    Starts(1,:)=[1,1,1];
    Stops(1,:)=[Thick,g.info.yy,g.info.zz];
    PEC(1,:)=[{'Ey'},{'Ez'},{'Hx'}];
    PMC(1,:)=[{'Hy'},{'Hz'},{'Ex'}];
    %% Update Second Border on X direction
    Starts(2,:)=[g.info.xx-Thick+1,1,1];
    Stops(2,:)=[g.info.xx,g.info.yy,g.info.zz];
    PEC(2,:)=[{'Ey'},{'Ez'},{'Hx'}];
    PMC(2,:)=[{'Hy'},{'Hz'},{'Ex'}];
    %% Update first border on Y direction
    Starts(3,:)=[1,1,1];
    Stops(3,:)=[g.info.xx,Thick,g.info.zz];
    PEC(3,:)=[{'Ex'},{'Ez'},{'Hy'}];
    PMC(3,:)=[{'Hx'},{'Hz'},{'Ey'}];
    %% Update Second Border on Y direction
    Starts(4,:)=[1,g.info.yy-Thick+1,1];
    Stops(4,:)=[g.info.xx,g.info.yy,g.info.zz];
    PEC(4,:)=[{'Ex'},{'Ez'},{'Hy'}];
    PMC(4,:)=[{'Hx'},{'Hz'},{'Ey'}];

    %% Update first border on Z direction
    Starts(5,:)=[1,1,1];
    Stops(5,:)=[g.info.xx,g.info.yy,Thick];
    PEC(5,:)=[{'Ex'},{'Ey'},{'Hz'}];
    PMC(5,:)=[{'Hx'},{'Hy'},{'Ez'}];
    %% Update Second Border on Z direction
    Starts(6,:)=[1,1,g.info.zz-Thick+1];
    Stops(6,:)=[g.info.xx,g.info.yy,g.info.zz];
    PEC(6,:)=[{'Ex'},{'Ey'},{'Hz'}];
    PMC(6,:)=[{'Hx'},{'Hy'},{'Ez'}];
    %% DX

    dX=Stops-Starts;


    [a,b]=size(Starts);
    temp=Bornum;
    for cnt=1:a

        if (Bornum(cnt)=='1') %Get the bit at the right position
            type;
            my_varargin=varargin;
            if strcmp(type,'perfectlayerzone')
                if strcmp(varargin{1},'PEC')
                    my_varargin=PEC(cnt,:);
                end

                if strcmp(varargin{1},'PMC')
                    my_varargin=PMC(cnt,:);
                end
                g=AddPerfectConductor(g,Starts(cnt,1),Starts(cnt,2),Starts(cnt,3),dX(cnt,1),dX(cnt,2),dX(cnt,3),my_varargin);
            end
        end


    end
end


end
