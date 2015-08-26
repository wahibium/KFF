find_package(logga REQUIRED)
message(STATUS "logga include dir: ${LOGGA_INCLUDE_DIR}")
message(STATUS "logga libraries dir: ${LOGGA_LIBRARIES}")
message(STATUS "logga version: ${LOGGA_VERSION_STRING}")
include_directories(${LOGGA_INCLUDE_DIR})
add_library(args.cc		graph.cc	main.cc		replace.cc	utils.cc
chromosome.cc	group.cc	mymath.cc	select.cc
fitness.cc	header.cc	operator.cc	stack.cc
getFileArgs.cc	help.cc		population.cc	startUp.cc
gga.cc		llist.cc	random.cc	statistics.cc)
