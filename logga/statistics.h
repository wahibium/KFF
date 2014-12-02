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

int intializeBasicStatistics(BasicStatistics *statistics, GGAParams *ggaParams);
int doneBasicStatistics(BasicStatistics *statistics);
int computeBasicStatistics(BasicStatistics *statistics, long t, Population *population, GGAAParams *ggaParams);

int generationStatistics(FILE *out, BasicStatistics *statistics);
int fitnessStatistics(FILE *out, BasicStatistics *statistics);
int finalStatistics(FILE *out, char *termination, BasicStatistics *statistics);
int printModel(FILE *out, long t, AcyclicOrientedGraph *G, FrequencyDecisionGraph **T);
int printGuidance(FILE *out, float *p1, int n, float treshold);

#endif
