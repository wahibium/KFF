// ################################################################################
//
// name:          replace.cc
//
// author:        Mohamed Wahib
//
// purpose:       the definition of replacement replacing the worst portion of the
//                original population and the divide and conquer function it uses
//                to separate the worst
//
// last modified: Feb 2014
//
// ################################################################################

#include "population.h"
#include "replace.h"
#include "random.h"

// ================================================================================
//
// name:          replaceTheWorst
//
// function:      performs the opposite to a truncation selection for replacement 
//                (replacing the worst guys from the population by the offspring)
//
// parameters:    population...the population where to put the offspring
//                parents......the offspring population
//
// returns:       (int) 0
//
// ================================================================================

int replaceWorst(Population *population, Population *offspring)
{
  long i,j;
  long M,N,NM;
  int n;

  // initialize variables

  N  = population->N;
  M  = offspring->N;
  n  = population->n;

  NM = N-M;

  // shuffle the individuals a little

  for (i=0; i<N; i++)
    {
      j = (long) ((double)drand()*(double)N);

      if (i!=j)
	swapIndividuals(population,i,j);
    };

  // separate worst M individuals by divide and conquer 

  divideWorst(population,0,N-1,n,N-M);

  // replace the worst M

  for (i=NM; i<N; i++)
    copyIndividual(population,i,offspring->x[i-NM],offspring->f[i-NM]);
  
  return 0;
}

// ================================================================================
//
// name:          divideWorst
//
// function:      do divide-and-conquer until the last M individuals (individuals
//                [NM...N]) are worse or equal than the rest (of totally N
//                individuals)
//
// parameters:    population...the population from which to separate the worst M
//                left.........a pointer pointing to the left-most individual of
//                             the currently processed part of the population
//                right........a pointer pointing to the right-most individual of
//                             the currently processed part of the population
//                NM...........size of the population minus the number of 
//                             individuals to separate (N-M)
//
// returns:       (int) 0
//
// ================================================================================

int divideWorst(Population *population, long left, long right, int n, long NM)
{
  long l,r;
  float pivot;

  l  = left;
  r  = right;

  pivot = (population->f[l]+population->f[r])/2;

  while (l<=r)
    {
      while ((l<right)&&(population->f[l]>pivot)) l++;
      while ((r>left)&&(population->f[r]<pivot)) r--;

      if (l<=r)
	{
	  if (l!=r)
	    swapIndividuals(population,l,r);

	  l++;
	  r--;
	}
    };

  if ((l==NM)||(r==(NM-1)))
    return 0;
  
  if ((r>=NM)&&(left<r))
    divideWorst(population,left,r,n,NM);

  if ((l<NM)&&(l<right))
    divideWorst(population,l,right,n,NM);
  
  return 0;
}

