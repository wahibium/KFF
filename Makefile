#  ===============================================================
#
#  Makefile for the end-to-End automation of GPU kernel 
#  fission/fusion to exploit coarse-grained locality
#
#  author:        Mohamed Wahib
#
#  last modified: Nov 2014
#
#  ===============================================================
#
#  usage:
#
#         make clean     	- cleans object files
#
#         make all       	- the same as make optimized
#		
#		  make logga 	 	- make only logga
#
#		  make metadata  	- make only metadataGatherer
#
#		  make graphs    	- make only graphsGenerator
#
#		  make translator	- make only translator
#
#         make tar.Z     	- create a .tar.Z archive of the files 
#                          	  for transfering the sources
#
# -----------------------------------------------------------------# ----------------------------------------------------------------------

# PROGRAM NAME
PROGRAM = run

# COMPILER
CC = g++

#FLAGS
FLAGS  = -O3 -std=c++98

# OBJECT FILES
OBJS  = args.o                   \      
        chromosome.cc 			 \
        fitness.o                \
        getFileArgs.o            \                
        gga.o                    \
		graph.o					 \
        group.o                  \
        header.o                 \
        help.o                   \
        llist.o 				 \
        main.o                   \
        mymath.o                 \
		operator.o               \
        population.o             \
        random.o                 \
        replace.o                \
        select.o                 \
        stack.o                  \
        startUp.o                \
        statistics.o             \
        utils.o 				 

# Specify folder
VPATH = logga 				\
		metadataGatherer 	\
		graphsGenerator		\
		translator

######################
.cc.o:
        $(CC) -c $(FLAGS) $?
.cpp.o:
        $(CC) -c $(FLAGS) $?
.CPP.o:
        $(CC) -c $(FLAGS) $? 

######################
all: $(OBJS)
	$(CC) -o $(PROGRAM) $(FLAGS)

######################
######################
clean: 
	@rm -f $(OBJS)

#
# make compressed tar file containing the source code
#

tar.Z:
	@tar cvf kff.tar *.cc *.h examples/input.* examples/output.* Makefile README COPYRIGHT ;
	@compress kff.tar
