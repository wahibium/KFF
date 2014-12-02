// ################################################################################
//
// name:          gga.cc      
//
// author:        Mohamed Wahib
//
// purpose:       functions for the initialization of the GGA, the GGA itself and 
//                a done method for the GGA
//
// last modified: Feb 2014
//
// ################################################################################

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

// --------------------
// various output files
// --------------------

FILE *logFile;
FILE *fitnessFile;
FILE *modelFile;

BasicStatistics populationStatistics;

// --------------------------------------------------------
// the description of termination criteria that are checked
// --------------------------------------------------------

char *terminationReasonDescription[5] = {
"No reason",
"Maximal number of generations reached",
"Solution convergence (with threshold epsilon)",
"Proportion of optima in a population reached the threshold",
"Optimum has been found"};


// ================================================================================
//
// name:          GGA
//
// function:      the kernel of the GGA (runs the GGA for a chosen problem)
//
// parameters:    ggaParams....the parameters sent to the GAA
//
// returns:       (int) 0
//
// ================================================================================

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
  

  // get back
  return 0;
}

// ================================================================================
//
// name:          terminationCriteria
//
// function:      checks whether some of the termination criteria wasn't matched
//                and returns the number of the criterion that has been met or 0
//
// parameters:    ggaParams....the parameters sent to the GGA
//
// returns:       (int) the number of a met criterion or 0 if none has been met
//
// ================================================================================

int terminationCriteria(GGAParams *ggaParams)
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

  // get back
  return result;
}

// ================================================================================
//
// name:          generateOffspring
//
// function:      generates offspring in the GGA (select candidates from population, and then uses them to generate offspring) 
//
// parameters:    t............the number of current generation
//                parents......the selected set of promising solutions
//                offspring....the resulting population of offspring
//                ggaParams....the parameters sent to the GGA
//
// returns:       (int) 0
//
// ================================================================================

int generateOffspring(long iteration, Population *parents, Population *offspring, GGAParams *ggaParams)
{

  // apply cross over
  

  // apply mutation at random



  return 0;
}

// ================================================================================
//
// name:          pause
//
// function:      waits for enter key if required
//
// parameters:    (none)
//
// returns:       (int) 0
//
// ================================================================================

int pause(GGAParams *ggaParams)
{
  if (ggaParams->pause)
    {
      printf("Press Enter to continue.");
      getchar();
    };

  return 0;
}

// ================================================================================
//
// name:          terminate
//
// function:      gets back to normal what was changed in initialization of the GGA
//
// parameters:    ggaParams....the parameters sent to the GGA
//
// returns:       (int) 0
//
// ================================================================================

int terminate(GGAParams *ggaParams)
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
   
  // get back

  return 0;  
}

// ================================================================================
//
// name:          getLogFile
//
// function:      returns a pointer to the log file strea
//
// parameters:    (none)
//
// returns:       (FILE*) a pointer to the log file stream
//
// ================================================================================

FILE *getLogFile()
{
  return logFile;
}

// ================================================================================
//
// name:          getModelFile
//
// function:      returns a pointer to the model file stream
//
// parameters:    (none)
//
// returns:       (FILE*) a pointer to the model file stream
//
// ================================================================================

FILE *getModelFile()
{
  return modelFile;
}

// ================================================================================
//
// name:          getFitnessFile
//
// function:      returns a pointer to the fitness file stream
//
// parameters:    (none)
//
// returns:       (FILE*) a pointer to the fitness file stream
//
// ================================================================================

FILE *getFitnessFile()
{
  return fitnessFile;
}
