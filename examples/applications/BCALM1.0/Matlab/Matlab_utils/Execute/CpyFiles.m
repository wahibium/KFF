%Copies the files with an array of string objects to the destination
%fcpy=[{'Quadrupole.m'},{'FabryPerotPlot.m'}];
%destination='glenfiddich:/home/pwahl/tmp';
% Assumes that the destination grants access with a public key/private key
% protocol.


function succes=CpyFiles(fcpy,destination,SimulInfo)

succes=0;
file=fopen('MyBash','w');
fprintf(file,'#!/bin/bash \n');
fprintf(file,  ['  FILETOCOPY=(']);
for i = (1:length(fcpy))
    
    fprintf(file,  [ '''' fcpy{i}  ''' ']);
end
fprintf(file, ')\n');
fprintf(file, '    for FILE in ${FILETOCOPY[@]}\n');
fprintf(file,' do\n');
fprintf(file,[' scp -P ' SimulInfo.Port ' $FILE ' destination '\n']);
fprintf(file,[' done\n']);
fclose(file);
!chmod +x ./MyBash
!./MyBash
!rm MyBash

succes=1;