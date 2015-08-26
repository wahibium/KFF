%% source.theha= angle(Xaxis,(Kx,Ky,0))
%% source.phi = angle(Zaxis,(Kx,Ky,Kz))


function g=AddGaussianBeam(g,source)

minfield=1e-3;

%% Initialization of the source to be added 

newsource.omega=source.omega;
newsource.mut=source.mut;
newsource.sigmat=source.sigmat;
newsource.dx=0;
newsource.dy=0;
newsource.dz=0;



vectorE=[source.Ex,source.Ey,source.Ez];% This vector is going to be projected onto the orthonormal to the propagation vector
vectorH=[source.Hx,source.Hy,source.Hz];% This vector is going to be projected onto the orthonormal to the propagation vector



%% Get a unit vector in the propagation distance.
K=[sin(source.phi)*cos(source.tetha),sin(source.phi)*sin(source.tetha),cos(source.phi)]';
K=K/norm(K);
%% Get orthogonal complement of the K vector
R=orthcomp(K);
R(:,1)=R(:,1)/norm(R(:,1));
R(:,2)=R(:,2)/norm(R(:,2));

%% Calculation of the projected field amplitudes onto the basisset normal
%% to the propagation direction.

Eweights=vectorE*R
Hweights=vectorH*R

%% Calculation of the field weights in the original cartensian basis

FieldE=Eweights(1)*R(:,1)+Eweights(2)*R(:,2);
FieldH=Hweights(1)*R(:,1)+Hweights(2)*R(:,2);

%% Go over all the faces.
facenames=[{'facex'},{'facey'},{'facez'}];
dim=[g.info.xx,g.info.yy,g.info.zz];

for face=(1:length(facenames))
    if isfield(source,facenames(face))
        start=ones(1,3);
        stop=dim;
        %% Get the starts and stops of the forloops that go over the face
        start(face)= getfield(source,facenames{face});
        stop(face)=getfield(source,facenames{face});
        
          for x=(start(1):stop(1))
            % cartesian distance between the center of the min  beamwaist and the x coordinate
            pos(x,1)=-index2distance(g,x,'x')+index2distance(g,source.x,'x');
         end
         
         for y=(start(2):stop(2))
            % cartesian distance between the center of the min  beamwaist and the x coordinate
            pos(y,2)=-index2distance(g,y,'y')+index2distance(g,source.y,'y');
         end
         
          for z=(start(3):stop(3))
            % cartesian distance between the center of the min  beamwaist and the x coordinate
            pos(z,3)=-index2distance(g,z,'z')+index2distance(g,source.z,'z');
          end

                    

        %% Preallocate the sourcearray for the grid to increase speed;
  
       g=preallocatesource(g,start,stop)   ;
          
        
        for x=(start(1):stop(1))
            % cartesian distance between the center of the min  beamwaist and the x coordinate
           newsource.x=x;
            for y=(start(2):stop(2))
                % cartesian distance between the center of the min
                % beamwaist and the y coordinate
                    newsource.y=y;
                for z=(start(3):stop(3))
                    % cartesian distance between the center of the min
                    % beamwaist and the z coordinate
                        newsource.z=z;
                        
                     posprime=[pos(x,1),pos(y,2),pos(z,3)];
                    Zprime=posprime*K; % Z coordiante in the rotated cilinder coordiantes;
                    Rprime=sqrt(sum((posprime*R).^2)); %% R coordinate in the rotated cilinder coordinates;

                    E=GetGaussian(Rprime,Zprime,source.omega,source.n,source.w0);
                    Etest(x,y,z)=real(FieldE(1)*E);

                    if abs(E)>minfield
                    
                        newsource.Ex=FieldE(1)*E;
                        newsource.Ey=FieldE(2)*E;
                        newsource.Ez=FieldE(3)*E;
                        newsource.Hx=FieldH(1)*E;
                        newsource.Hy=FieldH(2)*E;
                        newsource.Hz=FieldH(3)*E;
                        g=AddSource(g,newsource);
                    end


                end
            end

        end





% 
% E=Etest;
% E=E((start(1):stop(1)),(start(2):stop(2)),(start(3):stop(3)));
% E=squeeze(E);
% size(E)
% figure;
% p=gca
% mypcolor(p,cumsum(g.grid.x),cumsum(g.grid.y),real(E));
% shading interp
% axis equal



    end
end


end
