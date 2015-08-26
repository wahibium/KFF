#include <stdio.h>
#include "chromosome.h"
#include "memalloc.h"
#include "utils.h"


void InitChromosome(Chromosome *chromosome)
{
  chromosome->chromoLen = 0;
  //chromosome->chromoIdx = id;
  //chromosome->groups = (Group*) Calloc(len,sizeof(Group));
  //for(int i=0;i<len;i++)
    //  initGroup(groups+i, )
    //  copyGroup(groups[i],groups[i]);
  return;
}

void FreeChromosome(Chromosome *chromosome)
{
  for(int i=0;i< chromosome->chromoLen;i++)
    freeGroup(chromosome->(groups+i));
  chromosome->groups.clear();
  chromosome->groups.shrink_to_fit();  
    //Free(chromosome->groups[i]);
  //Free(chromosome);
  return;
}


void PrintChromosome(Chromosome *chromosome, FILE *out)
{
  
  if (out==NULL)
  {
    strcpy(errorMsg,"No output stream to print chromosome");
    err_exit();
  }

  for (int k=0; k<chromosome->chromoLen; k++)
  {
    fprintf(out,"\nGroup: %d\n",k);
    printGroup(chromosome->(groups+k),out);
  }
  return;
}

void CopyChromosome(Chromosome *src, Chromosome *dest)
{
  
  dest->chromoLen = src->chromoLen;
  dest->chromoIdx = src->chromoIdx;
  //dest->groups = (Group*) Calloc(*dest->chromoLen,sizeof(Group));
  // Allocate groups and copy groups in chromosome
  //for(int i=0;i<*dest->chromoLen;i++)  {
    //dest->groups[i].groupVal = (int*) Calloc(src->dest->groups[i].groupLen,sizeof(int));
    //for(int j=0;j<dest->groups[i].groupLen;j++) {
      //dest->groups[i].groupVal[j] = src->groups[i].groupVal[j];
  dest->groups.reservse(src->groups.size());
  for(int j=0;j<dest->chromoLen;j++)
    copyGroup(src->(groups+j), dest->(groups+j));  
  //copy(src->groups.begin(),src->groups.end(),back_inserter(dest->groups));  
    //}
  //}
  return;
}

void AddGrouptoChromosome(Chromosome *chromosome, Group *group)
{
  chromosome->groups.push_back(*group);
  chromosome->chromoLen = chromosome->chromoLen + 1;
  chromosome->chromoIdx = chromosome->chromoLen;
  return;
}

void RemoveGroupFromChromosome(Chromosome *chromosome, Group *group)
{
  
  for(int j=0;j<group->groupLen;j++)   
     removeOriginalKernelFromGroup(group, group->groupKernelsIDs[j]); 
  // recrusive removal?
  chromosome->groups.erase(std::remove(chromosome->groups.begin(),chromosome->groups.end(),*group),chromosome->groups.end());
  chromosome->chromoLen = chromosome->chromoLen - 1;
  return;

}

void AddEmptyGrouptoChromosome(Chromosome *chromosome)
{
  //  chromosome->groups.push_back(*group);
  //  chromosome->chromoLen = chromosome->chromoLen + 1;
  Group groupToAdd;
  groupToAdd.groupLen  = 0;
  groupToAdd.groupIdx  = chromosome->groups.size();
  chromosome->groups.push_back(&groupToAdd);
  chromosome->chromoLen = chromosome->chromoLen + 1;
  return;
}