// ################################################################################
//
// name:          random.cc
//
// author:        Mohamed Wahib
//
// purpose:       random number generator related functions (random generator is 
//                based on the code by Fernando Lobo, Prime Modulus Multiplicative 
//                Linear Congruential Generator (PMMLCG)
//
// last modified: Mohamed Wahib 2014
//
// ################################################################################

#include <stdio.h>
#include <math.h>

#include "random.h"

long _Q = _M/_A;     // M / A 
long _R = _M%_A;     // M mod A
long _seed;          // a number between 1 and m-1

//char whichGaussian=0; // which gaussian to generate

// ================================================================================
//
// name:          drand
//
// function:      returns a floating-point random number generated according to
//                uniform distribution from [0,1)
//
// parameters:    (none)
//
// returns:       (double) resulting random number
//
// ================================================================================

double drand()
{
  long lo,hi,test;
  
  hi   = _seed / _Q;
  lo   = _seed % _Q;
  test = _A*lo - _R*hi;
  
  if (test>0)
    _seed = test;
  else
    _seed = test+_M;

  return double(_seed)/_M;
}

// ================================================================================
//
// name:          intRand
//
// function:      returns an integer random from [0,max)
//
// parameters:    max..........the upper bound
//
// returns:       (int) resulting random number
//
// ================================================================================

int intRand(int max)
{
   return (int) ((double) drand()*max);
}

// ================================================================================
//
// name:          longRand
//
// function:      returns a long integer random from [0,max)
//
// parameters:    max..........the upper bound
//
// returns:       (long) resulting random number
//
// ================================================================================

long longRand(long max)
{
   return (long) ((double) drand()*(double) max);
}

// ================================================================================
//
// name:          flipCoin
//
// function:      returns 0 or 1 with the same probability
//
// parameters:    (none)
//
// returns:       (int) resulting number
//
// ================================================================================

char flipCoin()
{
  if (drand()<0.5)
      return 1;
  else
      return 0;
}

// ================================================================================
//
// name:          setSeed
//
// function:      sets the random seed
//
// parameters:    seed.........a new random seed
//
// returns:       (long) the result of the operation
//
// ================================================================================

long  setSeed(long newSeed)
{
  // set the seed and return the result of the operation
  
  return (_seed = newSeed);
}
