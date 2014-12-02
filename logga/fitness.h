#ifndef _FITNESS_H_
#define _FITNESS_H_

#include "gga.h"

typedef double FitnessFunction(char *x, int n);
typedef char  IsBest(char *x, int n);
typedef int   InitFitness(GGAParams *ggaParams);
typedef int   DoneFitness(GGAParams *ggaParams);

typedef struct {
  char            *description;

  FitnessFunction *fitness;
  IsBest          *isBest;
  InitFitness     *init;
  DoneFitness     *done;
} Fitness;


double roofline(Chromosome *chromosome);
double simplemodel(Chromosome *chromosome);
double complexmodel(Chromosome *chromosome);

char bestSolution(char *x, int n);


int  setFitness(int n);
char *getFitnessDesc(int n);

int initializeFitness(GGAParams *ggaParams);
int doneFitness(GGAParams *ggaParams);
double getFitnessValue(char *x, int n);
int isBestDefined();
int isOptimal(char *x, int n);

int resetFitnessCalls(void);
long fitnessCalled(void);
long getFitnessCalls(void);

#endif
