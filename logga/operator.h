#ifndef _OPERATOR_H_
#define _OPERATOR_H_

//#include "graph.h"
#include "chromosome.h"

#define OPERATOR_NONE               0 
#define OPERATOR_CROSSOVER_ONLY     1
#define OPERATOR_MUTATION_ONLY      2
//#define OPERATOR_CROSSOVER_MUTATION 3

//struct CrossoverOperator {
  //float gain;
  //int   type;
  //int   where;
  //int   label;

//};

//struct MutationOperator {
  //float gain;
  //LabeledTreeNode *a;
  //LabeledTreeNode *b;
//};
void applyCrossover(Population *parents, Population *offspring, GGAParams *ggaParams);
void applyMutation(Population *parents, Population *offspring, GGAParams *ggaParams);
//int resetOperator(Operator *x);
//int updateBestOperator(Operator *best, Operator *x);
//int updateBestNodeOperator(Operator *x, FrequencyDecisionGraph *t, MergeOperator *merge, int numMerges, int node, int n);
//int applyOperator(Operator *x, AcyclicOrientedGraph *G);
//int updateGainsAfterOperator(Operator *x, AcyclicOrientedGraph *G, int maxParents, long N);
//int operatorApplicable(Operator *x, AcyclicOrientedGraph *G);
//int deleteOperator(Operator *x);

#endif
