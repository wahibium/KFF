
Population::generatePopulation()
Population::isChromosomeFeasible (int Idx)
Population::initialize(BoaParams *boaParams)


@ GGA


int generateOffspring(Population *parents, Population *offspring, GGAParams *ggaParams)




generatePopulation
initializePopulation
Operators
select
replace
Obj. Function
Constraints
GGA
read param from file
stat.
------------------------
create D-graph
create O-graph
-----------------------
create metadata (instrumentation)
------------------------
fuse kernels with no halo
-----------------------
fuse two kernel with halo
-------------------------
create new CUDA code with invocation to new kernels
--------------------------
visualize the transformation process
---------------------------
wrap/test/release