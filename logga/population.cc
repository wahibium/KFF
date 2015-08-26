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

void Population::~Population()
{

  for(int j=0;j<len;j++)
    for(int i=0;i< chromosome->chromoLen;i++)
      Free(chromosomes[j]->groupArr[i]);
  Free(f);
  Free(chromosomes);
  return;
}

void Population::GeneratePopulation()
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

void Population::EvaluatePopulation()
{
  
  for (int i=0; i<N; i++)
    f[i] = EvaluateChromosome(chromosomes[i]);
  
  return;
}


double Population::EvaluateChromosome(int Idx)
{
  
  return GetFitnessValue(chromosomes[Idx]);
}


bool Population::IsChromosomeFeasible(int Idx)
{
  if(GetFitnessValue(chromosomes[Idx]) > 0)
    return true;
  else
    return false;
}


void Population::SwapGroups(int srcGroupIdx, int srcChromosomeIdx, int destGroupIdx, int destChromosomeIdx)
{

  // Allocate temp. Copy src to tempGroup
  Group tempGroup;
  AllocateGroup(tempGroup, (chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx)->groupLen, (chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx)->groupVal);
  
  // dest to src
  FreeGroup((chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx));  
  AllocateGroup((chromosomes+srcChromosomeIdx)->(groupArr+srcGroupIdx), (chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx)->groupLen, (chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx)->groupVal);  

  // tmp to dest
  FreeGroup((chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx));
  AllocateGroup((chromosomes+destChromosomeIdx)->(groupArr+destGroupIdx), tempGroup.groupLen, tempGroup.groupVal);  

  return;
}

void Population::PrintPopulation(FILE *out)
{

  if (out==NULL)
  {
    strcpy(errorMsg,"No output stream to print Population");
    err_exit();
  }

  for(int j=0;j<N;j++){
      fprintf(out,"\nChromosome: %d\n",j);
      PrintChromosome(chromosomes+j,out);
    }
  }
  return;

}

int Population::GetBestChromosomeID()
{
  double topFitnessValue = EvaluateChromosome(0);
  int topFitnessIdx = 0;
  for(int i=1;i<N)
    if(EvaluateChromosome(i)<topFitnessValue)
      {
        topFitnessValue = EvaluateChromosome(i);
        topFitnessIdx = i;
      }  
        
  return topFitnessIdx;

}

bool Population::Initialize(GGAParams *ggaParams)
{
  char filename[200];

  // set the fitness function to be optimized

  SetFitness(ggaParams->fitnessNumber);
 
  // initialize fitness

  InitializeFitness(ggaParams);

  // initialize metric

  InitializeMetric(ggaParams);

  // reset the counter for fitness calls

  ResetFitnessCalls();

  // set random seed

  SetSeed(ggaParams->randSeed);

  // initialize statistics

  IntializeBasicStatistics(&populationStatistics,ggaParams);

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

  return true;
}
