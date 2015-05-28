// ================================================================================
// ================================================================================
//
// Instructions for adding a new fitness function: 
// ------------------------------------------------
//
// 1. create a function with the same input parameters as other fitness functions 
//    defined in this file (e.g., onemax) that returns the value of the fitness
//    given a binary chromosome of a particular length (sent as input parameters
//    to the fitness)
//
// 2. put the function definition in the fitness.h header file (look at onemax
//    as an example)
//
// 3. increase the counter numFitness and add a structure to the array of the
//    fitness descriptions fitnessDesc below. For compatibility of recent input
//    files, put it at the end of this array, in order not to change the numbers 
//    of already defined and used functions. The structure has the following items
//    (in this order):
//      a) a string description of the function (informative in output data files)
//      b) a pointer to the function (simple "&" followed by the name of a function)
//      c) a pointer to the function that returns true if an input solution is
//         globally optimal and false if this is not the case. If such function is
//         not available, just use NULL instead. The algorithm will understand...
//      d) a pointer to the function for initialization of the particular fitness
//         function (not used in any of these and probably not necessary for most
//         of the functions, but in case reading input file would be necessary
//         or so, it might be used in this way). Use NULL if there is no such 
//         function
//      e) a pointer to the "done" function, called when the fitness is not to be
//         used anymore, in case some memory is allocated in its initialization;
//         here it can be freed. Use NULL if there is no need for such function
//
//  4. the function will be assigned a number equal to its ordering number in the
//     array of function descriptions fitnessDesc minus 1 (the functions are
//     assigned numbers consequently starting at 0); so its number will be equal
//     to the number of fitness definitions minus 1 at the time it is added. Its
//     description in output files will be the same as the description string
//     (see 3a)
//
// ================================================================================
// ================================================================================

#include <stdio.h>
#include <stdlib.h>

#include "chromosome.h"
#include "fitness.h"
#include "gga.h"

#define numFitness 3

static Fitness fitnessDesc[numFitness] = {
  {"Roofline Model",&roofline,&bestSolution,NULL,NULL},
  {"Simple Model",&simplemodel,&bestSolution,NULL,NULL},
  {"Complex Model",&complexmodel,&bestSolution,NULL,NULL},
};


Fitness *fitness;


long   fitnessCalls_;


char BestSolution(Chromosome *chromosome)
{
  register int i;

  for (i=0; (i<n)&&(x[i]==1); i++);
    
  return (i==n);
}

int SetFitness(int n)
{
  if ((n>=0) && (n<numFitness))
    fitness = &(fitnessDesc[n]);
  else
    {
      fprintf(stderr,"ERROR: Specified fitness function doesn't exist (%u)!",n);
      exit(-1);
    }
  
  return 0;
}


char* GetFitnessDesc(int n)
{
  return fitnessDesc[n].description;
}


int InitializeFitness(GGAParams *ggaParams)
{
  if (fitness->init)
    return fitness->init(ggaParams);

  return 0;
}

int DoneFitness(GGAParams *ggaParams)
{
  if (fitness->done)
    return fitness->done(ggaParams);
  
  return 0;
}

float GetFitnessValue(Chromosome *chromosome) 
{
  fitnessCalled();
  return fitness->fitness(x,n);
}

int IsBestDefined()
{
  return (int) (fitness->isBest!=NULL);
}


int IsOptimal(char *x, int n) 
{
 return fitness->isBest(x,n);
}

int ResetFitnessCalls(void)
{
  return (int) (fitnessCalls_=0);
}


long FitnessCalled(void)
{
  return fitnessCalls_++;
}


long GetFitnessCalls(void)
{
  return fitnessCalls_;
}
