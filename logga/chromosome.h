#ifndef _CHROMOSOME_H_
#define _CHROMOSOME_H_

#include "group.h"
#include <vector.h>

using namespace std;

struct Chromosome{

  int chromoIdx;		   // zero indexing
  int   chromoLen;         // chromosome length
  //Group *groupArr;       //  chromosome groups
  vector<Group> groups;	   //  groups in chromosome
};
  //void initChromosome(Chromosome *chromosome, int len, Group *gArr, int id);
  void initChromosome(Chromosome *chromosome);
  void freeChromosome(Chromosome *chromosome);
  void printChromosome(Chromosome *chromosome, FILE *out);    
  void copyChromosome(Chromosome *src, Chromosome *dest);
  void addGrouptoChromosome(Chromosome *chromosome, Group *group);
  void removeGroupFromChromosome(Chromosome *chromosome, Group *group);
  void addEmptyGrouptoChromosome(Chromosome *chromosome);
  //void replaceGroup(Group *group, int idX);



#endif
