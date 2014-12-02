// ################################################################################
//
// name:          utils.cc
//
// author:        Mohamed Wahib
//
// purpose:       functions use elsewhere for swapping values of the variables
//                of various data types, and some more miscellaneous utils
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>

#include "utils.h"

// ================================================================================
//
// name:          swapInt
//
// function:      swaps the two integers
//
// parameters:    a............a pointer to the first number to exchange
//                b............a pointer to the second number to exchange
//
// returns:       (int) 0
//
// ================================================================================

int swapInt(int *a, int *b)
{
  int aux;
  
  aux = *a;
  *a  = *b;
  *b  = aux;

  return 0;
}

// ================================================================================
//
// name:          swapLong
//
// function:      swaps the two long integers
//
// parameters:    a............a pointer to the first number to exchange
//                b............a pointer to the second number to exchange
//
// returns:       (int) 0
//
// ================================================================================

int swapLong(long *a, long *b)
{
  long aux;
  
  aux = *a;
  *a  = *b;
  *b  = aux;

  return 0;
}

// ================================================================================
//
// name:          swapPointers
//
// function:      swaps the two pointers
//
// parameters:    a............a pointer to the first pointer to exchange
//                b............a pointer to the second pointer to exchange
//
// returns:       (int) 0
//
// ================================================================================

void swapPointers(void **a, void **b)
{
  void *aux;

  aux = *a;
  *a  = *b;
  *b  = aux;
}

// ================================================================================
//
// name:          printSpaces
//
// function:      prints out a certain number of spaces
//
// parameters:    out..........a pointer to the output stream
//                num..........a number of spaces to print out
//
// returns:       (int) 0
//
// ================================================================================

int printSpaces(FILE *out, int num)
{
  int i;
 
  for (i=0; i<num; i++)
    fputc(' ',out);

  return 0;
}
