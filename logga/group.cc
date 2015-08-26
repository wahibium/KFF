#include <stdio.h>
#include <algorithm>
#include "group.h"
#include "memalloc.h"
#include "utils.h"


void InitGroup(Group* group, int len, int *originalKernelsIDs, int id)
{
  group->groupLen = len;
  group->groupIdx = id;
  //group->groupKernelsIDs = (int*) Calloc(len,sizeof(int));
  for(int i=0;i<len;i++)
      group->groupKernelsIDs.push_back(*(originalKernelsIDs+i));
  return;
}

void FreeGroup(Group* group)
{
  
  //Free(group->groupKernelsIDs);
  group->groupKernelsIDs.clear();
  group->groupKernelsIDs.shrink_to_fit();
  return;
}

void PrintGroup(Group* group, FILE *out)
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

void CopyGroup(Group *src, Group *dest)
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

void AddOriginalKernelToGroup(Group* group, int originalKernelId)
{

    group->groupKernelsIDs.push_back(originalKernelId);
    group->groupLen = group->groupLen + 1;
    return;

}

void RemoveOriginalKernelFromGroup(Group* group, int originalKernelId)
{

    group->groupKernelsIDs.erase(std::remove(group->groupKernelsIDs.begin(),group->groupKernelsIDs.end(),originalKernelId),group->groupKernelsIDs.end());
    group->groupLen = group->groupLen - 1;
    return;
}

/*
bool IsFeasibleToAddOriginalKernel(Group* group, int originalKernelId)
{

  // make sure there are sharing sets with at least one of the original kernels
  return true;
}
*/

bool IsOriginalKernelInGroup(Group* group, int originalKernelId)
{

  vector<int>::iterator it;
  it = find(group->groupKernelsIDs.begin(), group->groupKernelsIDs.end(), originalKernelId);
  if (it != groupKernelsIDs.end())
    return true;
  else
    return false;
  
}

bool IsTwoGroupsIntersecting(Group* groupOne, Group* groupTwo, Group* intersectionResult)
{
  vector<int>::iterator it;
  it = set_intersection (groupOne, groupOne + groupOne->groupLen, groupTwo, groupTwo + groupTwo->groupLen, v.begin());                                               
  intersectionResult->groupKernelsIDs.resize(it-v.begin());                      
  if(intersectionResult->groupKernelsIDs.size() == 0)
      return false;
  intersectionResult->groupLen = intersectionResult->groupKernelsIDs.size();
  return true;  
}

