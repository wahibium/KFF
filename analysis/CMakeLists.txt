# Boost
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED program_options)
include_directories(${Boost_INCLUDE_DIRS})

set(ANALYSIS_SRC activityFactor.cc    hammockGraphAnalysis.cc
                  affineAnalysis.cc   loopAnalysis.cc
                  analysis.cc     measureBranchDivergence.cc
                  analysisFactory.cc    measureMemEfficiency.cc
                  barrierCount.cc     measureThreadIC.cc
                  basicBlockExecutionCount.cc measureWarpIC.cc
                  branchDivergence.cc   memoryDependenceAnalysis.cc
                  branchInfo.cc     memoryEfficiency.cc
                  clockCycleCount.cc    postdominatorTree.cc
                  controlDependenceAnalysis.cc
                  controlTree.cc      programStructureGraph.cc
                  convergentRegionAnalysis.cc sSAGraph.cc
                  cycleAnalysis.cc    safeRegionAnalysis.cc
                  dataDependenceAnalysis.cc simpleAliasAnalysis.cc
                  dataflowGraph.cc    sparseAnalysis.cc
                  dependenceAnalysis.cc   structuralAnalysis.cc
                  directionalGraph.cc   superblockAnalysis.cc
                  divergenceAnalysis.cc   threadFrontierAnalysis.cc
                  divergenceGraph.cc    threadInstructionCount.cc
                  dominatorTree.cc    warpInstructionCount.cc)

if (CUDA_FOUND AND CUDA_RUNTIME_ENABLED)
  include_directories(${CUDA_INCLUDE_DIR})
  if (AUTO_TUNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif ()
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(
      kff_rt_cuda PROPERTIES
      COMPILE_FLAGS "-stdlib=libstdc++"
      LINK_FLAGS "-stdlib=libstdc++"
      )
  endif ()
endif()

if (CUDA_FOUND AND CUDA_HM_RUNTIME_ENABLED)
  include_directories(${CUDA_INCLUDE_DIR})
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(
      kff_rt_cuda_hm PROPERTIES
      COMPILE_FLAGS "-stdlib=libstdc++"
      LINK_FLAGS "-stdlib=libstdc++"
      )
  endif ()
  install(kff_rt_cuda_hm DESTINATION lib)
endif()

