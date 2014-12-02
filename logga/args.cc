//################################################################################
//
// name:          args.cc
//
// author:        Mohamed Wahib
//
// purpose:       functions for manipulation with arguments passed to a program
//
// last modified: Feb 2014
//
// ################################################################################

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "args.h"

// ================================================================================
//
// name:          isArg
//
// function:      checks whether there is a particular parameter in command line 
//                parameters
//
// parameters:    s............the parameter to look for
//                argc.........the number of arguments sent to the program
//                argv.........an array of arguments sent to the program (including
//                             its name and the path)
//
// returns:       (int) 1 if the parameter has been found, 0 otherwise
//
// ================================================================================

int isArg(char *s, int argc, char **argv)
{

  for (int i=0; i<argc; i++)  
      if (!strcmp(s,argv[i]))
	 return 1;

  return 0;
}
