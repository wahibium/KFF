// ################################################################################
//
//
// author:        Mohamed Wahib  <mohamed.attia@riken.jp>
//
//
// last modified: May 2015
//
// ################################################################################

#include <stdio.h>

#include "gga.h"
#include "getFileArgs.h"
#include "args.h"
#include "fitness.h"
#include "startUp.h"
#include "header.h"

GGAParams ggaParams;

// the definition of input parameters that can be specified in the input file 

ParamStruct params[] =
{
  {PARAM_LONG,"populationSize",&boaParams.N,"1200","Size of the population",NULL},
  {PARAM_FLOAT,"offspringPercentage",&boaParams.percentOffspring,"50","Size of offspring to create (% from population)",NULL},

  {PARAM_DIVIDER,NULL,NULL,NULL,NULL,NULL},  

  {PARAM_INT,"fitnessFunction",&boaParams.fitnessNumber,"2","Number of fitness function to use",&getFitnessDesc},
  {PARAM_INT,"problemSize",&boaParams.n,"30","Size of the problem (of one dimension)",NULL},

  {PARAM_DIVIDER,NULL,NULL,NULL,NULL,NULL},  

  {PARAM_INT,"tournamentSize",&boaParams.tournamentSize,"4","Tournament size (selection pressure)",NULL},
  {PARAM_LONG,"maxNumberOfGenerations",&boaParams.maxGenerations,"200","Maximal Number of Generations to Perform",NULL},
  {PARAM_LONG,"maxFitnessCalls",&boaParams.maxFitnessCalls,"-1","Maximal Number of Fitness Calls (-1 when unbounded)",NULL},
  {PARAM_FLOAT,"epsilon",&boaParams.epsilon,"0.01","Termination threshold for the univ. freq. (-1 is ignore)",NULL},
  {PARAM_CHAR,"stopWhenFoundOptimum",&boaParams.stopWhenFoundOptimum,"0","Stop if the optimum was found?", &yesNoDescriptor},
  {PARAM_FLOAT,"maxOptimal",&boaParams.maxOptimal,"-1","Percentage of opt. & nonopt. ind. threshold (-1 is ignore)",NULL},
 
  {PARAM_DIVIDER,NULL,NULL,NULL,NULL,NULL},  

  {PARAM_INT,"maxIncoming",&boaParams.maxIncoming,"20","Maximal number of incoming edges in dep. graph for the BOA",NULL},
  {PARAM_CHAR,"allowMerge",&boaParams.allowMerge,"0","Allow a merge operator?",&yesNoDescriptor},

  {PARAM_DIVIDER,NULL,NULL,NULL,NULL,NULL},

  {PARAM_CHAR,"pause",&boaParams.pause,"0","Wait for enter after printing out generation statistics?",NULL},

  {PARAM_DIVIDER,NULL,NULL,NULL,NULL,NULL},

  {PARAM_STRING,"outputFile",&(boaParams.outputFilename),NULL,"Output file name",NULL},
  {PARAM_FLOAT,"guidanceThreshold",&boaParams.guidanceThreshold,"0.3","Threshold for guidance (closeness to 0,1)",NULL},

  {PARAM_DIVIDER,NULL,NULL,NULL,NULL,NULL},

  {PARAM_LONG,"randSeed",&boaParams.randSeed,"time","Random Seed",NULL},
  
  {PARAM_END,NULL,NULL,NULL,NULL}
};


int main(int argc, char **argv)
{
  // process the arguments, read the input file (if specified)

  StartUp(argc, argv, params);

  // initialize LOGGA

  Initialize(&ggaParams);

  // print the header to stdout, and most of the output files

  PrintTheHeader(stdout);
  PrintTheHeader(getLogFile());
  PrintTheHeader(getModelFile());

  // print the values of the parameters to stdout and the log file

  PrintParamValues(stdout,params);
  PrintParamValues(getLogFile(),params);

  // run the boa

  GGA(&ggaParams);

  // free the used resources

  Done(&ggaParams);


  return 0;
}
