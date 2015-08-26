#include <stdio.h>
#include <stdlib.h>

#include "statistics.h"
#include "population.h"
#include "fitness.h"
#include "boa.h"
#include "mymath.h"
#include "memalloc.h"
#include "graph.h"
#include "frequencyDecisionGraph.h"


int IntializeBasicStatistics(BasicStatistics *statistics, BoaParams *boaParams)
{
  // allocate memory for univariate frequencies

  statistics->p1 = (float*) Calloc(boaParams->n,sizeof(float));

  // get back

  return 0;
}


int DoneBasicStatistics(BasicStatistics *statistics)
{
  // free the memory used by the univariate frequencies

  Free(statistics->p1)

  return 0;
}

int ComputeBasicStatistics(BasicStatistics *statistics, long t, Population *population, BoaParams *boaParams)
{
  long i;

  // set some variables

  statistics->N                 = population->N;
  statistics->n                 = population->n;
  statistics->generation        = t;
  statistics->guidanceThreshold = boaParams->guidanceThreshold;

  // compute the maximal, minimal and average fitness in the population

  statistics->max = 0;
  statistics->minF = statistics->maxF = statistics->avgF = population->f[0];
  if (isBestDefined())
    statistics->numOptimal = (isOptimal(population->x[0],statistics->n))? 1:0;

  for (i=1; i<statistics->N; i++)
    {
      statistics->avgF += population->f[i];

      if (population->f[i]<statistics->minF)
	statistics->minF = population->f[i];
      else
	if (population->f[i]>statistics->maxF)
	  {
	    statistics->maxF = population->f[i];
	    statistics->max = i;
	  }

      if (isBestDefined())
	if (isOptimal(population->x[i],statistics->n))
	  statistics->numOptimal++;
    };

  statistics->avgF /= (double) statistics->N;

  // allocate memory for and compute the univariate frequencies

  computeUnivariateFrequencies(population,statistics->p1);

  // set the best guy (if defined...)

  if (isBestDefined())
    statistics->bestX = population->x[statistics->max];


  return 0;
}


int GenerationStatistics(FILE *out, BasicStatistics *statistics)
{
  // output not null?

  if (out==NULL)
    return 0;

  // print it all out

  fprintf(out,"--------------------------------------------------------\n");
  fprintf(out,"Generation                   : %lu\n",statistics->generation);
  fprintf(out,"Fitness evaluations          : %lu\n",getFitnessCalls());
  fprintf(out,"Fitness (max/avg/min)        : (%5f %5f %5f)\n",statistics->maxF,statistics->avgF,statistics->minF);
  if (isBestDefined())
    fprintf(out,"Percentage of optima in pop. : %1.2f\n",((float)statistics->numOptimal/(float)statistics->N)*100);
  fprintf(out,"Population bias              : ");
  printGuidance(out,statistics->p1,statistics->n,statistics->guidanceThreshold);
  fprintf(out,"\n");
  fprintf(out,"Best solution in the pop.    : ");
  if (isBestDefined())
    printIndividual(out,statistics->bestX,statistics->n);
  else
    fprintf(out,"Statistic not available.");
  fprintf(out,"\n");

  return 0;
}


int FitnessStatistics(FILE *out, BasicStatistics *statistics)
{
  // output not null?

  if (out==NULL)
    return 0;

  // print it all out

  fprintf(out,"%3lu %7lu %10f %10f %10f\n",statistics->generation,getFitnessCalls(),statistics->maxF,statistics->avgF,statistics->minF);

  return 0;
}

int FinalStatistics(FILE *out, char *termination, BasicStatistics *statistics)
{
  // is output stream null?

  if (out==NULL)
    return 0;

  // print it all out

  fprintf(out, "\n=================================================================\n");
  fprintf(out, "FINAL STATISTICS\n");
  fprintf(out, "Termination reason           : %s\n",termination);
  fprintf(out, "Generations performed        : %lu\n",statistics->generation);
  fprintf(out, "Fitness evaluations          : %lu\n",getFitnessCalls());
  fprintf(out, "Fitness (max/avg/min)        : (%5f %5f %5f)\n",statistics->maxF,statistics->avgF,statistics->minF);
  if (isBestDefined())
    fprintf(out, "Percentage of optima in pop. : %1.2f\n",((float)statistics->numOptimal/(float)statistics->N)*100);
  fprintf(out,"Population bias              : ");
  printGuidance(out,statistics->p1,statistics->n,statistics->guidanceThreshold);
  fprintf(out,"\n");
  fprintf(out,"Best solution in the pop.    : ");
  if (isBestDefined())
    printIndividual(out,statistics->bestX,statistics->n);
  else
    fprintf(out,"Statistic not available.");
  fprintf(out,"\n\nThe End.\n");


  return 0;
}

int PrintModel(FILE *out, long t, AcyclicOrientedGraph *G, FrequencyDecisionGraph **T)
{
  int i;

  // is output stream null?

  if (out==NULL)
    return 0;

  // print out the generation number

  fprintf(out,"--------------------------------------------------------\n");
  fprintf(out,"Generation: %3lu\n\n",t);

  // print out the model

  for (i=0; i<G->size(); i++)
    {
      fprintf(out,"%3u:\n",i);
      T[i]->print(out,5);
    };

  // get back

  return 0;
}


int PrintGuidance(FILE *out, float *p1, int n, float threshold)
{
  int k;
  float threshold1;
  
  // compute upper threshold

  threshold1 = 1-threshold;

  // print where the frequencies are biased

  for (k=0; k<n; k++)
    if (p1[k]<threshold)
      fprintf(out,"0");
    else
      if (p1[k]>threshold1)
	fprintf(out,"1");
      else
	fprintf(out,".");

  // get back

  return 0;
}
