// ################################################################################
//
// name:          population.cc
//
// author:        Mohamed Wahib
//
// purpose:       functions for manipulation with the populations of strings and
//                the strings themselves
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>
#include <string.h>
#include <vector.h>
  
#include "population.h"
#include "random.h"
#include "memalloc.h"
#include "fitness.h"
#include "originalkernel.h"
#include "sharingset.h"

using namespace std;

// ================================================================================
//
// name:          Population
//
// function:      allocates memory for a population and sets its size
//
// parameters:    N............the number of chromosomes for a population to contain
//                
//
// returns:       void
//
// ================================================================================

void Population::Population(int len)
{

  N = len;
  chromosomes = (Chromosome*) Calloc(N,sizeof(Chromosome));

  f = (double*) Calloc(N,sizeof(double));
  for (int i=0;i<N;i++)
  {
    chromosomes[i].chromoLen = 0;
    chromosomes[i].chromoIdx = i;
  }
  return;
}

// ================================================================================
//
// name:          ~Population
//
// function:      frees the memory used by a population
//
// parameters:    
//
// returns:       void
//
// ================================================================================

void Population::~Population()
{

  for(int j=0;j<len;j++)
    for(int i=0;i< chromosome->chromoLen;i++)
      Free(chromosomes[j]->groupArr[i]);
  Free(f);
  Free(chromosomes);
  return;
}

// ================================================================================
//
// name:          generatePopulation
//
// function:      generates all chromosomes in a population at random
//
// parameters:    none
//
// returns:       void
//
// ================================================================================

void Population::generatePopulation()
{
  // TWO METHODS FOR INITIALIZATION
  // METHOD 1
  // Each choromosome has x% unfused original kernels. x in [5,30]. Remaining kernels are fused at a rate of z in [2,4] original kernels per new kernel
  /*
  Algorithm: Generate Population
  -for each choromosome 
    -choose (numOriginalKernels-(x))% of original kernels from bag
    -For each kernel K in chosen
      -set value of z
      -search sharing sets of K for z kernels to fuse
      -validate, if not valid repeat the previous step
      -fit new kernel group randomly to chromosome
      -remove the fused original kernels from bag
    -fit remaining original kernels in bag to chromosome randomly
  */
    // METHOD 2
  /*
  Algorithm: Generate Population
  - shuffle original kernels randmoly
  - For each kernel 
      - Fuse to the first group such that no constraints are violated
      - If no suck group exists, create a new group and insert the kernel into it
      
  */

  // use vector as a bag for original kernels
  int randIdx, spotID;
  vector<int> bagOriginalKernels(numOriginalKernels);
  //choromosome tempChromosome;
  //for (int k = 0; k < numOriginalKernels; k++)
    //bagOriginalKernels.push_back(originalKernels[k].originalKernelID);
  

  // for each choromosome in the population
  for (int i=0;i<N;i++)
  {
    //fill(bagOriginalKernels.begin(),numOriginalKernels,-1);  
    for (int k = 0; k < numOriginalKernels; k++)
      bagOriginalKernels.push_back(k+1);
    std::srand(i);
    random_shuffle(bagOriginalKernels.begin(), bagOriginalKernels.end());
    
    //x = intRand(26) + 5;       
    // add each original kernel to the current chromosome
    for (int j = 0; j < numOriginalKernels; j++)
    {
      spotID = 0;
      isSpotFound = false;
      while (spotID < chromosomes[i].chromoLen)
      {
         if (isFeasibleToAddOriginalKernel(&(chromosomes[i].groupArr[spotID]), bagOriginalKernels[j]))  
         {
           addOriginalKernelToGroup(&(chromosomes[i].groupArr[spotID]), bagOriginalKernels[j]);
           break;
         }
         spotID++; 

      }
      if (spotID == chromosomes[i].chromoLen)
      {
          addEmptyGrouptoChromosome(chromosomes[i]);
          //addGrouptoChromosome(Chromosome *chromosome, Group *group);
          addOriginalKernelToGroup(&(chromosomes[i].groupArr[spotID]), bagOriginalKernels[j]);
      }
      //randIdx = intRand(numOriginalKernels) + 1;
      //bagOriginalKernels[randIdx] = j;
      //z = intRand(3) + 2;
      //y = intRand(bagOriginalKernels.size());
      //bagOriginalKernels.at(y);
      
      // Find a valid sharing set containing original kernel with ID bagOriginalKernels.at(y)
      //SharingSet 
      //int numSharingSets;

    } // original kernel loop


  } // chromosome loop
    
  

  return;
}

// ================================================================================
//
// name:          evaluatePopulation
//
// function:      evaluates fitness for all chromosomes in the population
//
// parameters:    none
//
// returns:       void
//
// ================================================================================

void Population::evaluatePopulation()
{
  
  for (int i=0; i<N; i++)
    f[i] = evaluateChromosome(chromosomes[i]);
  
  return;
}

// ================================================================================
//
// name:          evaluateChromosome
//
// function:      evaluates fitness for a chromosome in the population
//
// parameters:    Idx......index of chromosome to evaluate
//
// returns:       double
//
// ================================================================================

double Population::evaluateChromosome(int Idx)
{
  
  return getFitnessValue(chromosomes[Idx]);
}

// ================================================================================
//
// name:          isChromosomeFeasible
//
// function:      checks if a chromosome is a feasible solution
//
// parameters:    Idx......index of chromosome to check
//
// returns:       bool
//
// ================================================================================

bool Population::isChromosomeFeasible(int Idx)
{
  //getFitnessValue(chromosomes[Idx])
  
  return true;
}

// ================================================================================
//
// name:          swapGroups
//
// function:      swaps specified two groups in different chromosomes in the population
//
// parameters:    srcGroupIdx..........the population where to swap the specified string
//                srcChromosomeIdx.....a position of the first string to swap
//                destGroupIdx.........a position of the second string to swap
//                destChromosomeIdx....a position of the second string to swap
//
// returns:       void
//
// ================================================================================

void Population::swapGroups(int srcGroupIdx, int srcChromosomeIdx, int destGroupIdx, int destChromosomeIdx)
{

  // Allocate temp. Copy src to tempGroup
  Group tempGroup;
  allocateGroup(tempGroup, (chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx)->groupLen, (chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx)->groupVal);
  
  // dest to src
  freeGroup((chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx));  
  allocateGroup((chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx), (chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx)->groupLen, (chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx)->groupVal);  

  // tmp to dest
  freeGroup((chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx));
  allocateGroup((chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx), tempGroup.groupLen, tempGroup.groupVal);  

  return;
}

// ================================================================================
//
// name:          printPopulation
//
// function:      prints out the poplulation to a stream
//
// parameters:    out..........the stream to print the to
//
// returns:       void
//
// ================================================================================

void Population::printPopulation(FILE *out)
{

  if (out==NULL)
  {
    strcpy(errorMsg,"No output stream to print Population");
    err_exit();
  }

  for(int j=0;j<N;j++){
      fprintf(out,"\nChromosome: %d\n",j);
      printChromosome(chromosomes+j,out);
    }
  }
  return;

}

// ================================================================================
//
// name:          getBestChromosomeID
//
// function:      return ID of chromosome with best fitness
//
// parameters:    none
//
// returns:       int
//
// ================================================================================
int Population::getBestChromosomeID()
{
  double topFitnessValue = evaluateChromosome(0);
  int topFitnessIdx = 0;
  for(int i=1;i<N)
    if(evaluateChromosome(i)<topFitnessValue)
      {
        topFitnessValue = evaluateChromosome(i);
        topFitnessIdx = i;
      }  
        
  return topFitnessIdx;

}

// ================================================================================
//
// name:          initialize
//
// function:      initializes everything the BOA needs to be run properly
//                (initialize fitness function, metric, random number generator,
//                etc.)
//
// parameters:    boaParams....the parameters sent to the BOA
//
// returns:       bool
//
// ================================================================================

bool Population::initialize(GGAParams *ggaParams)
{
  char filename[200];

  // set the fitness function to be optimized

  setFitness(ggaParams->fitnessNumber);
 
  // initialize fitness

  initializeFitness(ggaParams);

  // initialize metric

  initializeMetric(ggaParams);

  // reset the counter for fitness calls

  resetFitnessCalls();

  // set random seed

  setSeed(ggaParams->randSeed);

  // initialize statistics

  intializeBasicStatistics(&populationStatistics,ggaParams);

  // open output files (if the base of the output file names specified)

  if (ggaParams->outputFilename)
    {
      sprintf(filename,"%s.log",ggaParams->outputFilename);
      logFile = fopen(filename,"w");

      sprintf(filename,"%s.fitness",ggaParams->outputFilename);
      fitnessFile = fopen(filename,"w");

      sprintf(filename,"%s.model",ggaParams->outputFilename);
      modelFile = fopen(filename,"w");
    }
  else
    logFile = fitnessFile = modelFile = NULL;

  // get back

  return true;
}
