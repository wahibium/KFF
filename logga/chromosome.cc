// ################################################################################
//
// name:          chromosome.cc
//
// author:        Mohamed Wahib
//
// purpose:       functions for manipulation with the chromosomes in a population (each chromosome represents a solution)
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>
#include "chromosome.h"
#include "memalloc.h"
#include "utils.h"

// ================================================================================
//
// name:          initChromosome
//
// function:      initialized a chromosome, sets its size to zero
//
// parameters:    chromosome.....chromosome to initialize
//
// returns:       void
//
// ================================================================================
void initChromosome(Chromosome *chromosome)
{
  chromosome->chromoLen = 0;
  //chromosome->chromoIdx = id;
  //chromosome->groups = (Group*) Calloc(len,sizeof(Group));
  //for(int i=0;i<len;i++)
    //  initGroup(groups+i, )
    //  copyGroup(groups[i],groups[i]);
  return;
}
  
// ================================================================================
//
// name:          freeChromosome
//
// function:      frees the memory used by the chromosome
//
// parameters:    chromosome.....chromosome to free
//
// returns:       void
//
// ================================================================================

void freeChromosome(Chromosome *chromosome)
{
  for(int i=0;i< chromosome->chromoLen;i++)
    freeGroup(chromosome->(groups+i));
  chromosome->groups.clear();
  chromosome->groups.shrink_to_fit();  
    //Free(chromosome->groups[i]);
  //Free(chromosome);
  return;
}


// ================================================================================
//
// name:          printChromosome
//
// function:      prints out a chromosome to a stream
//
// parameters:    chromosome.....chromosome to print
//                out..........the stream to print the string to
//
// returns:       void
//
// ================================================================================


void printChromosome(Chromosome *chromosome, FILE *out)
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

// ================================================================================
//
// name:          copyChromosome
//
// function:      copies a chromosome into another
//
// parameters:    src........group to copy
//                dest.......group to be copied into 
//
// returns:       void
//
// ================================================================================
void copyChromosome(Chromosome *src, Chromosome *dest)
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

// ================================================================================
//
// name:          addGrouptoChromosome
//
// function:      adds a new group to the chromosome
//
// parameters:    chromosome...chromosome that will include the group
//                group........group to add
//
// returns:       void
//
// ================================================================================
void addGrouptoChromosome(Chromosome *chromosome, Group *group)
{
  chromosome->groups.push_back(*group);
  chromosome->chromoLen = chromosome->chromoLen + 1;
  chromosome->chromoIdx = chromosome->chromoLen;
  return;
}

// ================================================================================
//
// name:          removeGroupFromChromosome
//
// function:      removes a group from the chromosome
//
// parameters:    chromosome...chromosome that will include the group
//                group........group to remove
//
// returns:       void
//
// ================================================================================
void removeGroupFromChromosome(Chromosome *chromosome, Group *group)
{
  
  for(int j=0;j<group->groupLen;j++)   
     removeOriginalKernelFromGroup(group, group->groupKernelsIDs[j]); 
  // recrusive removal?
  chromosome->groups.erase(std::remove(chromosome->groups.begin(),chromosome->groups.end(),*group),chromosome->groups.end());
  chromosome->chromoLen = chromosome->chromoLen - 1;
  return;

}

// ================================================================================
//
// name:          addEmptyGrouptoChromosome
//
// function:      adds a new group to the chromosome
//
// parameters:    chromosome...chromosome that will include the group
//
// returns:       void
//
// ================================================================================
void addEmptyGrouptoChromosome(Chromosome *chromosome)
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