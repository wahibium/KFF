// ################################################################################
//
// name:          mymath.cc
//
// author:        Mohamed Wahib
//
// purpose:       commonly used mathematical functions
//
// last modified: Feb 2014
//
// ################################################################################

#include "mymath.h"
#include "memalloc.h"

// ================================================================================
//
// name:          round
//
// function:      rounds a float
//
// parameters:    x............the input floating-point number
//  
// returns:       (long) integer closest to the input number
//
// ================================================================================

long round(float x)
{
  return (long) (x+0.5);
}
