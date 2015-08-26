#ifndef _POPULATION_H_
#define _POPULATION_H_

#include <stdio.h>
#include "gga.h"
#include "chromosome.h"
#include "utils.h"


class Population {

  long  N;        				// population size
  Chromosome *chromosomes;		// Chromosomes in the population
  double *f;       // fitness values

  public:
  Population(int len);
  ~Population();
  void PrintPopulation();
  void GeneratePopulation();
  void EvaluatePopulation();
  int AddChromosome(long where, Chromosome addedChromosome);  
  double EvaluateChromosome(int Idx);
  bool IsChromosomeFeasible(int Idx);
  void SwapGroups(int srcGroupIdx, int srcChromosomeIdx, int destGroupIdx, int destChromosomeIdx);
  int GetBestChromosomeID();  
  bool Initialize(GGAParams *params);
};

//int copyIndividual(Population *population, long where, char *x, float f);
//int swapIndividuals(Population *population, long first, long second);
//int printIndividual(FILE *out, char *x, int n);

#endif
