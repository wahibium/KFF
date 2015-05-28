#include <stdio.h>
#include <stdlib.h>

#include "gga.h"
#include "population.h"
#include "statistics.h"
#include "replace.h"
#include "select.h"
#include "graph.h"
#include "fitness.h"
#include "memalloc.h"
#include "random.h"

// various output files

FILE *logFile;
FILE *fitnessFile;
FILE *modelFile;

BasicStatistics populationStatistics;


// the description of termination criteria that are checked

char *terminationReasonDescription[5] = {
"No reason",
"Maximal number of generations reached",
"Solution convergence (with threshold epsilon)",
"Proportion of optima in a population reached the threshold",
"Optimum has been found"};


int GGA(GGAParams *ggaParams)
{
  long       N,numOffspring,numParents,t;
  // set sizes of populations
  N            = ggaParams->N;
  numOffspring = (long) ((float) ggaParams->N*ggaParams->percentOffspring)/100;
  numParents   = (long) ggaParams->N;

  Population population(N),parents(numParents),offspring(numOffspring);
  int        terminationReason;

  // initialize the poplulation according to parameters
  population.initialize(ggaParams);

  // randomly generate first population according to uniform distribution
  population.generatePopulation();

  // evaluate first population
  population.evaluatePopulation();

  // main loop
  t=0;

  // compute basic statistics on initial population
  computeBasicStatistics(&populationStatistics,t,&population,ggaParams);

  // output the statistics on first generation
  generationStatistics(stdout,&populationStatistics);
  generationStatistics(logFile,&populationStatistics);
  fitnessStatistics(fitnessFile,&populationStatistics);

  // pause after statistics?
  pause(ggaParams);

  while (!(terminationReason=terminationCriteria(ggaParams)))
    {
      // perform truncation (block) selection
      selectTheBest(&population,&parents,ggaParams);

      // create offspring
      generateOffspring(t,&parents,&offspring,ggaParams);
      
      // evaluate the offspring
      offspring.evaluatePopulation();

      // replace the worst of the population with offspring
      replaceWorst(&population,&offspring);
      
      // increase the generation number
  
      t++;

      // compute basic statistics
      computeBasicStatistics(&populationStatistics,t,&population,ggaParams);

      // output the statistics on current generation
      generationStatistics(stdout,&populationStatistics);
      generationStatistics(logFile,&populationStatistics);
      fitnessStatistics(fitnessFile,&populationStatistics);

      // pause after statistics?
      pause(ggaParams);
    };

  // print out final statistics
  computeBasicStatistics(&populationStatistics,t,&population,ggaParams);
  
  finalStatistics(stdout,terminationReasonDescription[terminationReason],&populationStatistics);
  finalStatistics(logFile,terminationReasonDescription[terminationReason],&populationStatistics);
  
  return 0;
}


int TerminationCriteria(GGAParams *ggaParams)
{
  int result;

  // no reason to finish yet

  result=0;

  // check if the proportion of optima reached the required value, if yes terminate
  if ((!result)&&(ggaParams->maxOptimal>=0))
    result = (float(populationStatistics.numOptimal*100)/(float)populationStatistics.N>=ggaParams->maxOptimal)? MAXOPTIMAL_TERMINATION:0;

  // check if should terminate if optimum has been found and if this is the case if yes
  if ((!result)&&(ggaParams->stopWhenFoundOptimum))
    if (isBestDefined())
      result = (isOptimal(populationStatistics.bestX,populationStatistics.n))? OPTIMUMFOUND_TERMINATION:0;

  // if there's no reason to finish yet and the epsilon threshold was set, check it
  if ((!result)&&(ggaParams->epsilon>=0))
    {
      int   k;
      float epsilon1;

      // set epsilon1 to (1-ggaParams->epsilon)
      epsilon1 = 1-ggaParams->epsilon;

      // are all frequencies closer than epsilon to either 0 or 1?
      result=EPSILON_TERMINATION;
      for (k=0; k<populationStatistics.n; k++)
	if ((populationStatistics.p1[k]>=ggaParams->epsilon)&&(populationStatistics.p1[k]<=epsilon1))
	  result=0;
    }

  // check if the number of generations wasn't exceeded
  if ((!result)&&(ggaParams->maxGenerations>=0))
    result = (populationStatistics.generation>=ggaParams->maxGenerations)? MAXGENERATIONS_TERMINATION:0;

  return result;
}

/*
int GenerateOffspring(long iteration, Population *parents, Population *offspring, GGAParams *ggaParams)
{

  // apply cross over  

  // apply mutation at random

  return 0;
}
*/

int Pause(GGAParams *ggaParams)
{
  if (ggaParams->pause)
    {
      printf("Press Enter to continue.");
      getchar();
    };

  return 0;
}


int Terminate(GGAParams *ggaParams)
{
  // get rid of the metric

  doneMetric();

  // get rid of the fitness

  doneFitness(ggaParams);

  // statistics done

  doneBasicStatistics(&populationStatistics);

  // close output streams

  if (logFile)
    fclose(logFile);

  if (fitnessFile)
    fclose(fitnessFile);
 
 if (modelFile)
    fclose(modelFile);

  return 0;  
}


FILE *GetLogFile()
{
  return logFile;
}

FILE *GetModelFile()
{
  return modelFile;
}

FILE *GetFitnessFile()
{
  return fitnessFile;
}
