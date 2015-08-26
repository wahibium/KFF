find_package(device REQUIRED)
message(STATUS "device include dir: ${DEVICE_INCLUDE_DIR}")
message(STATUS "device libraries dir: ${DEVICE_LIBRARIES}")
message(STATUS "device version: ${DEVICE_VERSION_STRING}")
include_directories(${DEVICE_INCLUDE_DIR})
add_library(cudaDriver.cc		cudaRuntime.cc		extractedDeviceState.cc
cudaDriverApi.cc	cudaRuntimeInterface.cc	fatBinaryContext.cc
cudaDriverFrontend.cc	cudaWorkerThread.cc
cudaDriverInterface.cc	cuda_runtime.cc)
