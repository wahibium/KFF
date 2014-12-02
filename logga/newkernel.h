#ifndef _NEWKERNEL_H_
#define _NEWKERNEL_H_

#define NewIDoffset 2000

struct NewKernel
{
	/* data */
	int newKernelID;				// New Kernels IDs starts from 2001
	char *newKernelName;
	int *indicesOriginalKernels;	// Indices of original kernels fused
	// metadata
	float T;						// Projected Runtime (Sec)
	float Toriginal					// Original runtime (Sec): the sum of runtimes of original kernels
	int Fl;							// Number of FLOP (sum of FLOP in original kernel)
	int numMemoryOperations;		// Number of GMEM operations

};

#endif