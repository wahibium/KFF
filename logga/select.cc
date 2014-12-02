// ################################################################################
//
// name:          select.cc
//
// author:        Mohamed Wahib
//
// purpose:       the definition of truncation selection and the divide and conquer
//                function it uses to separate the best
//
// last modified: Feb 2014
//
// ################################################################################

#include "population.h"
#include "select.h"
#include "random.h"

// ================================================================================
//
// name:          selectTheBest
//
// function:      performs tournament selection (realizing tournaments on the guys)
//
// parameters:    population...the population where to select from
//                parents......the population where to put the selected to
//                params.......the parameters passed to the GGA
//
// returns:       (int) 0
//
// ================================================================================

int selectTheBest(Population *population, Population *parents, GGAParams *params)
{
  register long i,j;
  long N;
  long picked;
  long max;
  double maxF;

  // initialize some variables

  N = population->N;

  for (i=0; i<N; i++)
    {
      // perform a tournament

      picked = longRand(N);
      maxF   = population->f[picked];
      max    = picked;

      for (j=1; j<params->tournamentSize; j++)
	     {
	       picked = longRand(N);
	       if (population->f[picked]>maxF)
	         {
	           maxF = population->f[picked];
	           max  = picked;
	         }
	     }

      picked=max;

      // insert the picked guy into the selected population
      
      copyIndividual(parents,i,population->x[picked],population->f[picked]);
    }

  // get back

  return 0;
}
