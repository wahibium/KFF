#ifndef _ORIGINALKERNEL_H_
#define _ORIGINALKERNEL_H_
#define INPUT-ONLY 		1
#define IN-OUT-STRICT	2
#define IN-OUT-EXPAND	3
#define OUTPUT-ONLY		4

struct OriginalKernel
{
	// Data
	int originalKernelID;		// Original Kernal IDs range from 1~1000
	char *originalKernelName;
	int *ArrayIndices;			// Indices of arrays used in this kernel
	int *arrayIntention;		// Intention of used Arrays
	// Kernel metadata
	int BlocksSMX;
	int TB;
	int Thr;
	int B;
	int RT;
	int RAdr;
	float T;
	int Fl;
	int OrdShr;
	int Flop;
	int ShrLst;
	int Hal;
	int numMemoryOperations;	// Number of GMEM operations

};

OriginalKernel *originalKernels;
int numOriginalKernels;

#endif