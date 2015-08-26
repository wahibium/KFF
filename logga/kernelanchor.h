#ifndef _KERNELANCHOR_H_
#define _KERNELANCHOR_H_

#include "newkernel.h"
#include "dataarray.h"

struct KernelAnchor
{
	/* data */
	NewKernel *sharingKernels;
	DataArray *sharedArray;
	int numAnchoredArrays;
};


#endif
