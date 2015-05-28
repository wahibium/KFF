#include <stdio.h>

#include "header.h"
#include "startUp.h"


int PrintTheHeader(FILE *out)
{
  // is output stream non-null?

  if (out==NULL)
    return 0;

  // print the header to the output file

  fprintf(out,"\n====================================================\n");
  fprintf(out," LOGGA: (L)ocality (O)ptimization (G)rouped (G)enetic (A)lgorithm\n");
    fprintf(out,"\n");
  fprintf(out," Version 0.1 (Released in Feb 2015)\n");
  fprintf(out," Copyright (c) 2015 HPC Programming Framework Research Team, RIKEN Advanced Institute for Computational Science\n");
  fprintf(out," Author: Mohamed Wahib mohamed.attia@riken.jp\n");
  fprintf(out,"-----------------------------------------------\n");
  fprintf(out," Parameter values from: %s\n",(getParamFilename())? getParamFilename():"(default)");
  fprintf(out,"===============================================\n");
  fprintf(out,"\n");

  // get back

  return 0;
}
