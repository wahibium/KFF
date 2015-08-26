# CUDA_INCLUDE_PATH
# CUDA_LIBRARIES
# CUDA_FOUND = true if ROSE is found

find_package (CUDA)

if (CUDA_FOUND) 
  find_path (CUDA_INCLUDE_DIR
    NAMES CL/cl.h
    PATHS ${CUDA_INCLUDE_DIRS}
    )
  if (CUDA_INCLUDE_DIR)
    find_library(CUDA_LIBRARY
      NAMES cuda
      PATHS env LD_LIBRARY_PATH
      )
  endif ()
endif()

set (CUDA_FOUND FALSE)
if (CUDA_INCLUDE_DIR AND OPENCL_LIBRARY)
  message (STATUS "cuda found")  
  message (STATUS "CUDA_INCLUDE_DIR=${CUDA_INCLUDE_DIR}")
  message (STATUS "CUDA_LIBRARY=${CUDA_LIBRARY}")
  set (CUDA_FOUND TRUE)
else ()
  message (STATUS "cuda not found")
endif ()

MARK_AS_ADVANCED(
CUDA_INCLUDE_DIR
CUDA_LIBRARY
CUDA_FOUND
)
