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


char BestSolution(char *x, int n);


int  SetFitness(int n);
char *GetFitnessDesc(int n);

int InitializeFitness(GGAParams *ggaParams);
int DoneFitness(GGAParams *ggaParams);
double GetFitnessValue(char *x, int n);
int IsBestDefined();
int IsOptimal(char *x, int n);

int ResetFitnessCalls(void);
long FitnessCalled(void);
long GetFitnessCalls(void);

#endif
