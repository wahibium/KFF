#include <stdio.h>
#include <math.h>

#include "random.h"

long _Q = _M/_A;     // M / A 
long _R = _M%_A;     // M mod A
long _seed;          // a number between 1 and m-1

//char whichGaussian=0; // which gaussian to generate


double Drand()
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

int IntRand(int max)
{
   return (int) ((double) drand()*max);
}

long LongRand(long max)
{
   return (long) ((double) drand()*(double) max);
}

char FlipCoin()
{
  if (drand()<0.5)
      return 1;
  else
      return 0;
}

long  SetSeed(long newSeed)
{
  // set the seed and return the result of the operation
  
  return (_seed = newSeed);
}
