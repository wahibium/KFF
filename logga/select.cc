#include "population.h"
#include "select.h"
#include "random.h"

int SelectTheBest(Population *population, Population *parents, GGAParams *params)
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

  return 0;
}
