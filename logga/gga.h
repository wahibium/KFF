#ifndef _GGA_H_
#define _GGA_H_

#define NO_TERMINATION              0
#define MAXGENERATIONS_TERMINATION  1
#define EPSILON_TERMINATION         2
#define MAXOPTIMAL_TERMINATION      3
#define OPTIMUMFOUND_TERMINATION    4

#include "population.h"

typedef struct {

  long N;                      // population size
  int percentOffspring;        // size of offspring (in %) [FROM FLOAT TO INT]
  
  int fitnessNumber;           // number of fitness function to use
  int nMax;                       // Max size of a problem (length of a string = total number of original kernels)

  int   tournamentSize;        // size of the tournament (selection)
  long  maxGenerations;        // maximal number of generations to continue
  long  maxFitnessCalls;       // maximal number of fitness calls to continue
  char  stopWhenFoundOptimum;  // stop when the optimum has been found?
  float maxOptimal;            // maximal proportion of optimal solutions to continue

  float crossoverProbability;  // the probability of crossover
  float mutationProbability;   // the probability of mutation

  char *outputFilename;        // the name of ouput file
  float guidanceThreshold;     // the threshold for guidance in statistic info

  char pause;                  // wait for enter after printing out generation statistics?

  long randSeed;               // random seed

} GGAParams;

// ---------------------------------------


int GGA(Params *params);
int TerminationCriteria(GGAParams *params);
int Selection(GGAParams *params);
int GenerateOffspring(Population *parents, Population *offspring, GGAParams *params);
int Pause(GGAParams *params);
int Terminate(GGAParams *params);

FILE *GetLogFile();
FILE *GetModelFile();
FILE *GetFitnessFile();

#endif
