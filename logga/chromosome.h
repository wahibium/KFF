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
  void InitChromosome(Chromosome *chromosome);
  void FreeChromosome(Chromosome *chromosome);
  void PrintChromosome(Chromosome *chromosome, FILE *out);    
  void CopyChromosome(Chromosome *src, Chromosome *dest);
  void AddGrouptoChromosome(Chromosome *chromosome, Group *group);
  void RemoveGroupFromChromosome(Chromosome *chromosome, Group *group);
  void AddEmptyGrouptoChromosome(Chromosome *chromosome);
  //void replaceGroup(Group *group, int idX);



#endif
