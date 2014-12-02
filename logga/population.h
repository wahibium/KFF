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
  void printPopulation();
  void generatePopulation();
  void evaluatePopulation();
  int addChromosome(long where, Chromosome addedChromosome);  
  double evaluateChromosome(int Idx);
  bool isChromosomeFeasible(int Idx);
  void swapGroups(int srcGroupIdx, int srcChromosomeIdx, int destGroupIdx, int destChromosomeIdx);
  int getBestChromosomeID();  
  bool initialize(GGAParams *params);
};

//int copyIndividual(Population *population, long where, char *x, float f);
//int swapIndividuals(Population *population, long first, long second);
//int printIndividual(FILE *out, char *x, int n);

#endif
