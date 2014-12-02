// ################################################################################
//
// name:          help.cc
//
// author:        Mohamed Wahib
//
// purpose:       help (arguments description, input file parameters description)
//
// last modified: Feb 2014
//
// ################################################################################

#include "help.h"
#include "getFileArgs.h"

// ================================================================================
//
// name:          help
//
// function:      prints out help (either command line parameters description or 
//                input file parameters description)
//
// parameters:    what.........the type of help 
//                             (0...general command line parameters)
//                             (1...input file parameters description)
//                params.......an array of ParamStruct items (that are to printed
//                             out the description of eventually)
//
// returns:       (int) 0
//
// ================================================================================

int help(char what, ParamStruct *params)
{
  if (what==0)
    {
      printf("Command line parameters:\n");
      printf("-h                   display this help screen\n");
      printf("<filename>           configuration file name\n");
      printf("-paramDescription    print out the description of all parameters in configuration files\n");
    }
  else
    {
      printParamsDescription(stdout,params);
    }

  return 0;
}
