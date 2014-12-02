// ################################################################################
//
// name:          header.cc      
//
// author:        Mohamed Wahib
//
// purpose:       prints out the header saying the name of the product, its author,
//                the date of its release, and the file with input parameters (if 
//                any)
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>

#include "header.h"
#include "startUp.h"

// ================================================================================
//
// name:          printTheHeader
//
// function:      prints out the product name, its author, its version, and the
//                date of its realease
//
// parameters:    out..........output stream
//
// returns:       (int) 0
//
// ================================================================================

int printTheHeader(FILE *out)
{
  // is output stream non-null?

  if (out==NULL)
    return 0;

  // print the header to the output file

  fprintf(out,"\n====================================================\n");
  fprintf(out," LOGGA: (L)ocality (O)ptimization (G)rouped (G)enetic (A)lgorithm\n");
    fprintf(out,"\n");
  fprintf(out," Version 0.1 (Released in November 2014)\n");
  fprintf(out," Copyright (c) 2014 HPC Programming Framework Research Team, RIKEN Advanced Institute for Computational Science\n");
  fprintf(out," Author: Mohamed Wahib mohamed.attia@riken.jp\n");
  fprintf(out,"-----------------------------------------------\n");
  fprintf(out," Parameter values from: %s\n",(getParamFilename())? getParamFilename():"(default)");
  fprintf(out,"===============================================\n");
  fprintf(out,"\n");

  // get back

  return 0;
}
