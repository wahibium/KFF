#ifndef _SHARINGSET_H_
#define _SHARINGSET_H_

#include "originalkernel.h"
#include "dataarray.h"

struct SharingSet
{
	/* data */
	
	DataArray *sharedArray;			// Array that the kernels touch
	OriginalKernel *sharingKernels;	// Kernel touching the array
	int degreeSharing;				// Number of kernels in the sharing set
	int *orderSharing;				// Number of elements accessed by a thread in each kernel for the current array
};

SharingSet *sharingSets;
int numSharingSets;

#endif
