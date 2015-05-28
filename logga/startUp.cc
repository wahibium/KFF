#include <stdlib.h>

#include "startUp.h"
#include "boa.h"
#include "getFileArgs.h"
#include "args.h"
#include "help.h"


// the name of the input file

char *paramFilename;


int StartUp(int argc, char **argv, ParamStruct *params)
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

  GetParamsFromFile(paramFilename,params);

  // get back

  return 0;
}


char *GetParamFilename()
{
  return paramFilename;
}
