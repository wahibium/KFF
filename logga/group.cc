// ################################################################################
//
// name:          group.cc
//
// author:        Mohamed Wahib
//
// purpose:       functions for manipulation with the groups in chromosome (each group represents a new kernel)
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>
#include <algorithm>
#include "group.h"
#include "memalloc.h"
#include "utils.h"

// ================================================================================
//
// name:          initGroup
//
// function:      initializes a group, sets its size and set the values.
//
// parameters:    group.................group to initialize
//                len...................the size of the group
//                originalKernelsIDs....the IDs of the original kernel in the group
//                id....................id to assign to the group
//
// returns:       void
//
// ================================================================================

void initGroup(Group* group, int len, int *originalKernelsIDs, int id)
{
  group->groupLen = len;
  group->groupIdx = id;
  //group->groupKernelsIDs = (int*) Calloc(len,sizeof(int));
  for(int i=0;i<len;i++)
      group->groupKernelsIDs.push_back(*(originalKernelsIDs+i));
  return;
}
  
// ================================================================================
//
// name:          freeGroup
//
// function:      frees the memory used by the group
//
// parameters:    group........group to free
//
// returns:       void
//
// ================================================================================

void freeGroup(Group* group)
{
  
  //Free(group->groupKernelsIDs);
  group->groupKernelsIDs.clear();
  group->groupKernelsIDs.shrink_to_fit();
  return;
}


// ================================================================================
//
// name:          printGroup
//
// function:      prints out a group to a stream
//
// parameters:    group........group to print
//                out..........the stream to print the group to
//
// returns:       void
//
// ================================================================================


void printGroup(Group* group, FILE *out)
{
  
  if (out==NULL)
  {
    strcpy(errorMsg,"No output stream to print group");
    err_exit();
  }

  for (k=0; k<group->groupLen; k++)
    fprintf(out,"%d\n",group->groupKernelsIDs[k]);

  return;
}

// ================================================================================
//
// name:          copyGroup
//
// function:      copies a group into another
//
// parameters:    src........group to copy
//                dest.......group to be copied into 
//
// returns:       void
//
// ================================================================================
void copyGroup(Group *src, Group *dest)
{
  
  dest->groupLen = src->groupLen;
  dest->groupIdx = src->groupIdx;
  //*dest->groupKernelsIDs = (int*) Calloc(*dest->groupLen,sizeof(int));
  dest->groupKernelsIDs.reservse(src->groupKernelsIDs.size());
  //for(int i=0;i<*dest->groupLen;i++)
    //  *dest->groupKernelsIDs[i] = src->groupKernelsIDs[i];
  copy(src->groupKernelsIDs.begin(),src->groupKernelsIDs.end(),back_inserter(dest->groupKernelsIDs));
  return;
}

// ================================================================================
//
// name:          addOriginalKernelToGroup
//
// function:      add an original kernel to a group
//
// parameters:    group........group to which the original kernel is added
//                originalKernelId.......ID of original kernel
//
// returns:       void
//
// ================================================================================
void addOriginalKernelToGroup(Group* group, int originalKernelId)
{

    group->groupKernelsIDs.push_back(originalKernelId);
    group->groupLen = group->groupLen + 1;
    return;

}

// ================================================================================
//
// name:          removeOriginalKernelFromGroup
//
// function:      remove an original kernel from a group
//
// parameters:    group........group from which the original kernel is removed
//                originalKernelId.......ID of original kernel
//
// returns:       void
//
// ================================================================================
void removeOriginalKernelFromGroup(Group* group, int originalKernelId)
{

    group->groupKernelsIDs.erase(std::remove(group->groupKernelsIDs.begin(),group->groupKernelsIDs.end(),originalKernelId),group->groupKernelsIDs.end());
    group->groupLen = group->groupLen - 1;
    return;
}

// ================================================================================
//
// name:          isFeasibleToAddOriginalKernel
//
// function:      checks if an original kernel can be added to the group
//
// parameters:    group........group to check
//                originalKernelId.......ID of original kernel to be checked
//
// returns:       bool
//
// ================================================================================
bool isFeasibleToAddOriginalKernel(Group* group, int originalKernelId)
{

  // make sure there are sharing sets with at least one of the original kernels


  return true;
}

// ================================================================================
//
// name:          isOriginalKernelInGroup
//
// function:      checks if an original kernel is in a group
//
// parameters:    group........group to check
//                originalKernelId.......ID of original kernel to be checked
//
// returns:       bool
//
// ================================================================================
bool isOriginalKernelInGroup(Group* group, int originalKernelId)
{

  vector<int>::iterator it;
  it = find(group->groupKernelsIDs.begin(), group->groupKernelsIDs.end(), originalKernelId);
  if (it != groupKernelsIDs.end())
    return true;
  else
    return false;
  
}

// ================================================================================
//
// name:          isTwoGroupsIntersecting
//
// function:      checks two group have at least one OriginalKernel repeated in both
//
// parameters:    groupOne.....................group one to check
//                groupTwo.....................group two to check
//                Group* intersectionResult....group holding result of intersection
//
// returns:       bool
//
// ================================================================================
bool isTwoGroupsIntersecting(Group* groupOne, Group* groupTwo, Group* intersectionResult)
{
  vector<int>::iterator it;
  it = set_intersection (groupOne, groupOne + groupOne->groupLen, groupTwo, groupTwo + groupTwo->groupLen, v.begin());                                               
  intersectionResult->groupKernelsIDs.resize(it-v.begin());                      
  if(intersectionResult->groupKernelsIDs.size() == 0)
      return false;
  intersectionResult->groupLen = intersectionResult->groupKernelsIDs.size();
  return true;  
}

