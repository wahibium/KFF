find_package(meta REQUIRED)
message(STATUS "meta include dir: ${META_INCLUDE_DIR}")
message(STATUS "meta libraries dir: ${META_LIBRARIES}")
message(STATUS "meta version: ${META_VERSION_STRING}")
include_directories(${META_INCLUDE_DIR})
add_library(activeTimer.cc      memoryTraceAnalyzer.cc
activityFactorGenerator.cc  memoryTraceGenerator.cc
argumentParser.cc   parallelismTraceAnalyzer.cc
basicBlockCountGenerator.cc parallelismTraceGenerator.cc
branchTraceAnalyzer.cc    performanceBoundGenerator.cc
branchTraceGenerator.cc   sharedComputationAnalyzer.cc
cacheSimulator.cc   sharedComputationGenerator.cc
compression.cc      statisticDatabase.cc
configurable.cc     string.cc
controlFlowVisualizer.cc  systemCompatibility.cc
convergenceGenerator.cc   temporalSIMTGenerator.cc
debug.cc      test.cc
exception.cc      
floatingPoint.cc    timer.cc
instructionTraceAnalyzer.cc traceConfiguration.cc
instructionTraceGenerator.cc  warpSynchronousGenerator.cc
json.cc       watchTraceGenerator.cc
kernelDimensionsGenerator.cc  x86TraceGenerator.cc
kernelEntry.cc      xmlArgumentParser.cc
loadBalanceGenerator.cc   xmlLexer.cc
lowLevelTimer.cc    xmlParser.cc
machineAttributesGenerator.cc xmlTree.cc
math.cc)
