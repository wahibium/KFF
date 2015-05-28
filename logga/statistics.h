#ifndef _STATISTICS_H_
#define _STATISTICS_H_

#include <stdio.h>

#include "population.h"
#include "gga.h"
#include "graph.h"
#include "frequencyDecisionGraph.h"

// ----------------
// basic statistics
// ----------------

typedef struct {

  long   generation;       // the number of generation
  long   N;                // population size
  int    n;                // problem size (number of variables)
  float  minF,maxF;        // minimal and maximal fitness
  double avgF;             // average fitness
  long   numOptimal;       // number of optimal solutions
  long   max;              // number of maximal individual
  float  *p1;              // univariate frequencies
  char   *bestX;           // best candidate
  float guidanceThreshold; // guidance threshold

} BasicStatistics;

int IntializeBasicStatistics(BasicStatistics *statistics, GGAParams *ggaParams);
int DoneBasicStatistics(BasicStatistics *statistics);
int ComputeBasicStatistics(BasicStatistics *statistics, long t, Population *population, GGAAParams *ggaParams);

int GenerationStatistics(FILE *out, BasicStatistics *statistics);
int FitnessStatistics(FILE *out, BasicStatistics *statistics);
int FinalStatistics(FILE *out, char *termination, BasicStatistics *statistics);
int PrintModel(FILE *out, long t, AcyclicOrientedGraph *G, FrequencyDecisionGraph **T);
int PrintGuidance(FILE *out, float *p1, int n, float treshold);

#endif
