#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "args.h"

int isArg(char *s, int argc, char **argv)
{

  for (int i=0; i<argc; i++)  
      if (!strcmp(s,argv[i]))
	 return 1;

  return 0;
}
