function PovRay(SimulInfo)%% Sends my file to the PovRayRenderingmachine

system(['rm ' SimulInfo.PovRayPlace 'PovBasic.txt'])
system(['cp ' SimulInfo.file ' ' SimulInfo.PovRayPlace])
system([SimulInfo.PovRayPlace 'CreatePovRay.o -device=0 ' '''' SimulInfo.PovRayPlace 'FileBInM'''])

 system(['povray +I' SimulInfo.PovRayPlace 'PovBasic.txt +P +W800 +H800'])


