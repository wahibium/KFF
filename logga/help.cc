#include "help.h"
#include "getFileArgs.h"

int Help(char what, ParamStruct *params)
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
