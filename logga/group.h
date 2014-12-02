#ifndef _GROUP_H_
#define _GROUP_H_

#include <vector.h>
#define true 1
#define false 0

using namespace std;

 struct Group{

  int groupIdx;							// zero indexing
  int   groupLen;        				// Group length
  //int   *groupKernelsIDs;       		// IDs of original kernels in Group
  vector<int> groupKernelsIDs;	        // IDs of original kernels in Group

};

//int copyIndividual(Population *population, long where, char *x, float f);
//int swapIndividuals(Population *population, long first, long second);
void initGroup(Group* group, int len, int *originalKernelsIDs, int id);
void freeGroup(Group* group);
void copyGroup(Group *src, Group *dest);
void printGroup(Group* group, FILE *out);
void addOriginalKernelToGroup(Group* group, int originalKernelId);
void removeOriginalKernelFromGroup(Group* group, int originalKernelId);
bool isFeasibleToAddOriginalKernel(Group* group, int originalKernelId);
bool isOriginalKernelInGroup(Group* group, int originalKernelId);
bool isTwoGroupsIntersecting(Group* groupOne, Group* groupTwo, Group* intersectionResult);

#endif
