%Plots the value of a point trough time.Index is the Index in the array of
%Returned values. Field is a string with the field to plot, point is the
%coordinate of the point of interests. If several points are given those
%plots are superimposed.
%g is passed to get the outputs info

function PlotPointTime(g,Returned,index,field,point) 

t=g.outputs{index}.t;%*g.info.dt;
tt=g.outputs{index}.tt;

X=Returned{index}.field{GetField(field)}(point(1),point(2),point(3),1:end);
X=reshape(X,tt,1);
figure
plot(t,X);
title(['Time Evolution of point ' num2str(point)])
xlabel('Time')
ylabel(field)

