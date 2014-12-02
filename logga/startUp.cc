// ################################################################################
//
// name:          startUp.cc
//
// author:        Mohamed Wahib
//
// purpose:       a start-up function for processing the arguments passed to the
//                program and the function returning the name of the input file if
//                any was used
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdlib.h>

#include "startUp.h"
#include "boa.h"
#include "getFileArgs.h"
#include "args.h"
#include "help.h"

// --------------------------
// the name of the input file
// --------------------------

char *paramFilename;

// ================================================================================
//
// name:          startUp
//
// function:      processes the arguments passed to the program and calls the 
//                function for reading input file or help if required
//
// parameters:    argc.........the number of arguments sent to the program
//                argv.........an array of arguments sent to the program (including
//                             its name and the path)
//                params.......an array of ParamStruct items (that are to be either
//                             read from an input file or set to their default
//                             values)
//
// returns:       (int) 0
//
// ================================================================================

int startUp(int argc, char **argv, ParamStruct *params)
{
  // help requested?

  if (isArg("-h",argc,argv))
     {
       help(0,params);
       exit(0);
     }

  // description of parameters requested?

  if (isArg("-paramDescription",argc,argv))
    {
      help(1,params);
      exit(0);
    }

  // too many arguments?

  if (argc>2) 
    {
      fprintf(stderr,"ERROR: Too many arguments.\n");
      help(0,params);
      exit(-1);
    };

  // read the paramters file name and read it all

  if (argc==2)
    paramFilename = argv[1];
  else
    paramFilename = NULL;

  getParamsFromFile(paramFilename,params);

  // get back

  return 0;
}

// ================================================================================
//
// name:          getParamFilename
//
// function:      returns the file name of the used input file (or NULL)
//
// parameters:    (none)
//
// returns:       (char*) the file name of the used input file or NULL if defaults
//                have been used
//
// ================================================================================

char *getParamFilename()
{
  return paramFilename;
}
