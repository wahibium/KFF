% Create the basic, underlying grid structure
%
% grid = create_grid (dims, tt)
%
% DIMS should be a 3-element vector describing the dimensions of the simulation space in the following order [XX YY ZZ].
% TT is the amount of time to run the simulation
% GRID is a matlab structure that contains all the information describing and defining the simulation.

function [grid] = create_grid (dims,dt,tt)
% hard coded in
T = 0; % starting point for time

% put in those important things 
grid.info = struct ( 'xx', dims(1), 'yy', dims(2), 'zz', dims(3), 'dt', dt, 'tt', tt, 'T', T);


