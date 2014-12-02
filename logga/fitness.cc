// ################################################################################
//
// name:          fitness.cc
//
// author:        Mohamed Wahib
//
// purpose:       the definition of fitness functions; in order to add a fitness  
//                one has to add it here (plus the definition in the header file
//                fitness.h); see documentation or the instructions below
//
// last modified: Feb 2014
//
// ################################################################################


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

// ------------------
// the fitness in use
// ------------------

Fitness *fitness;

// ---------------------------
// the number of fitness calls
// ---------------------------

long   fitnessCalls_;

// ================================================================================
//
// name:          simplemodel
//
// function:      computes the expected runtime by removing the measured memory access time
//
// parameters:    chromosome............ solution to be evaluated
//
// returns:       (float) runtime in sec
//
// ================================================================================

double simplemodel(Chromosome *chromosome)
{
  double f = 1.0;
  return f;
}

// ================================================================================
//
// name:          complexmodel
//
// function:      computes the expected runtime according to model extending Lau. et al
//
// parameters:    chromosome............ solution to be evaluated
//
// returns:       (float) runtime in sec
//
// ================================================================================

double complexmodel(Chromosome *chromosome)
{
  double f = 1.0;
  return f;
}

// ================================================================================
//
// name:          roofline
//
// function:      computes the expected runtime according to the Roofline model
//
// parameters:    chromosome............ solution to be evaluated
//
// returns:       (float) runtime in sec
//
// ================================================================================
double roofline(Chromosome *chromosome)
{
   double f= 1.0;
   return f; 
}
// ================================================================================
//
// name:          bestSolution
//
// function:      checks whether input is the best known solution
//
// parameters:    chromosome............ solution to be evaluated
//
// returns:       (char) non-zero if best solution, 0 otherwise
//
// ================================================================================

char bestSolution(Chromosome *chromosome)
{
  register int i;

  for (i=0; (i<n)&&(x[i]==1); i++);
    
  return (i==n);
}



// ================================================================================
// ================================================================================
// ================================================================================


// ================================================================================
//
// name:          setFitness
//
// function:      sets the tested function according to its ordering number
//
// parameters:    n............the number of a fitness to use
//
// returns:       (int) 0
//
// ================================================================================

int setFitness(int n)
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

// ================================================================================
//
// name:          getFitnessDesc
//
// function:      gets a string description of a particular function
//
// parameters:    n............the number of a fitness to get the description of
//
// returns:       (char*) the string description of the function
//
// ================================================================================

char *getFitnessDesc(int n)
{
  return fitnessDesc[n].description;
}

// ================================================================================
//
// name:          initializeFitness
//
// function:      calls an init method of chosen fitness (if any)
//
// parameters:    ggaParams....the parameters passed to the GGA
//
// returns:       (int) return value of the called method or 0
//
// ================================================================================

int initializeFitness(GGAParams *ggaParams)
{
  if (fitness->init)
    return fitness->init(ggaParams);

  return 0;
}

// ================================================================================
//
// name:          doneFitness
//
// function:      calls a done method of chosen fitness (if any)
//
// parameters:    boaParams....the parameters passed to the BOA
//
// returns:       (int) return value of the called method or 0
//
// ================================================================================

int doneFitness(GGAParams *ggaParams)
{
  if (fitness->done)
    return fitness->done(ggaParams);
  
  return 0;
}

// ================================================================================
//
// name:          getFitnessValue
//
// function:      evaluates the fitness for an input string
//
// parameters:    chromosome............ solution to be evaluated
//
// returns:       (double) the value of chosen fitness for the input string
//
// ================================================================================

float getFitnessValue(Chromosome *chromosome) 
{
  fitnessCalled();
  return fitness->fitness(x,n);
}

// ================================================================================
//
// name:          isBestDefined
//
// function:      checks whether the proposition identifying the optimal strings is
//                defined for a chosen function
//
// parameters:    (none)
//
// returns:       (int) non-zero if the proposition function defined, 0 otherwise
//
// ================================================================================

int isBestDefined()
{
  return (int) (fitness->isBest!=NULL);
}

// ================================================================================
//
// name:          isOptimal
//
// function:      checks whether the input string is optimal (assuming the 
//                proposition function is defined)
//
// parameters:    x............the string
//                n............the length of the string
//
// returns:       (int) the value of the proposition
//
// ================================================================================

int isOptimal(char *x, int n) 
{
 return fitness->isBest(x,n);
}

// ================================================================================
//
// name:          resetFitnessCalls
//
// function:      resets the number of fitness calls (sets it to 0)
//
// parameters:    (none)
//
// returns:       (int) 0
//
// ================================================================================

int resetFitnessCalls(void)
{
  return (int) (fitnessCalls_=0);
}

// =============================================================

long fitnessCalled(void)
{
  return fitnessCalls_++;
}

// =============================================================

long getFitnessCalls(void)
{
  return fitnessCalls_;
}
