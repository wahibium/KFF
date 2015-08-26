#include "population.h"
#include "replace.h"
#include "random.h"

int ReplaceWorst(Population *population, Population *offspring)
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


int DivideWorst(Population *population, long left, long right, int n, long NM)
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

