% export the grid to an hdf file
%
% hdf_export (filename, grid)
%
% FILENAME is the name of the file you want to export the grid to
% GRID is the grid

function [] = hdf_export (filename, grid)

fname = {'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'};
cname = {'x', 'y', 'z'};


%% Update the info Fields

grid=UpdateInfo(grid);


filename;




%% export the grid

names = fieldnames (grid.grid);
dset_details.Location = '/in/grid';
for cnt = 1 : length (names)
    if (cnt == 1)
        wmode = 'overwrite';
    else
        wmode = 'append';
    end
	dset = getfield (grid.grid, names{cnt});
	dset = single (dset);
	dset_details.Name=names{cnt};
	hdf5write (filename, dset_details, dset, 'WriteMode', wmode);
end

%% Export deps

%% Get deps parameters
tempeps=[];
tempsigma=[];
for cnt=(1:length(grid.deps))
    tempeps=[tempeps grid.deps(cnt).epsilon];
    tempsigma=[tempsigma grid.deps(cnt).sigma];
end

dset = single (tempeps);
dset_details.Location = '/parameters';
dset_details.Name = 'deps';
wmode = 'append';
hdf5write (filename, dset_details,dset, 'WriteMode', wmode);

dset = single (tempsigma);
dset_details.Location = '/parameters';
dset_details.Name = 'sigma';
wmode = 'append';
hdf5write (filename, dset_details,dset, 'WriteMode', wmode);
%% Get dielzones
if isfield(grid,'dielzone')
    if (length(grid.dielzone)>0)
    tempzones=getzones(grid,'dielzone');
    else
        tempzones=0;
    end
else
    tempzones=0;
    warning('No dielectrics defined');
end
dset = int32 (tempzones);
dset_details.Location = '/in';
dset_details.Name = 'dielzone';
wmode = 'append';
hdf5write (filename, dset_details,dset, 'WriteMode', wmode);

%% Export the lorentz cells.
if isfield(grid,'lorentz')
    temp=getzones(grid,'lorentz');
else
    temp=0;
    warning('No Lorentz cells defined')
end
    dset = single (temp);
    dset_details.Location = '/parameters';
    dset_details.Name = 'lorentz';
    wmode = 'append';
    hdf5write (filename, dset_details,dset, 'WriteMode', wmode);

    %% Get lorentzzones
    if isfield(grid,'lorentzzone')
    tempzones=getzones(grid,'lorentzzone');
    else
        tempzone=0;
        warning('No Lorentzzones defined');
    end
    
    dset = int32 (tempzones);
    dset_details.Location = '/in';
    dset_details.Name = 'lorentzzone';
    wmode = 'append';
    hdf5write (filename, dset_details,dset, 'WriteMode', wmode);

    %% Export the the sources
    if isfield(grid,'source')
    dset_details.Location = '/parameters';
    dset_details.Name = 'source';
    grid.source=grid.source(:,(1:grid.info.ss));
    hdf5write (filename, dset_details, single (grid.source), 'WriteMode', 'append');
    end
    %% Export the the perfectlayerzones
    if isfield(grid,'perfectlayerzone')
        tempzones=getzones(grid,'perfectlayerzone');
        dset = int32 (tempzones);
        dset_details.Location = '/in';
        dset_details.Name = 'perfectlayerzone';
        wmode = 'append';
        hdf5write (filename, dset_details, dset, 'WriteMode', wmode);
    end

    %% Export the the cpmlzone
    if isfield(grid,'cpmlzone')
        tempzones=getzones(grid,'cpmlzone');
        dset = single (tempzones);
        dset_details.Location = '/in';
        dset_details.Name = 'cpmlzone';
        wmode = 'append';
        hdf5write (filename, dset_details, dset, 'WriteMode', wmode);
    end

    %% Export my output list
    if isfield(grid,'outputzones')
        tempzones=getzones(grid,'outputzones');
        dset = single (tempzones);
        dset_details.Location = '/in';
        dset_details.Name = 'outputzones';
        wmode = 'append';
        hdf5write (filename, dset_details, dset, 'WriteMode', wmode);
        %% Write the string arrays to give names to the objects.
        tempzones=getzones(grid,'outputzonesname');
         dset = tempzones;
        dset_details.Location = '/in';
        dset_details.Name = 'outputzonesname';
        wmode = 'append';
        hdf5write (filename, dset_details, dset, 'WriteMode', wmode);
        

    end





    %% export all the supporting information
    names = fieldnames (grid.info);
    dset_details.Location = '/info';
    for cnt = 1 : length (names)
        dset = getfield (grid.info, names{cnt});
        % appropriately choose between storing as float or as int
        if ( names{cnt}(1) == 'd')
            dset = single (dset);
        else
            dset = int32 (dset);
        end
        dset_details.Name = names{cnt};
        hdf5write (filename, dset_details, dset, 'WriteMode', 'append');
    end



