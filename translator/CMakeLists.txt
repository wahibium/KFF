find_package(trans REQUIRED)
message(STATUS "trans include dir: ${TRANS_INCLUDE_DIR}")
message(STATUS "trans libraries dir: ${TRANS_LIBRARIES}")
message(STATUS "trans version: ${TRANS_VERSION_STRING}")
include_directories(${TRANS_INCLUDE_DIR})
add_library(ast.cc      common.cc   range.cc
attribute.cc    config.cc   reference.cc
aux.cc      kernel.cc   stencil.cc
ccuda.cc    optimizations.cc  translator.cc
Macintosh-5:translator wahib$ 
)
