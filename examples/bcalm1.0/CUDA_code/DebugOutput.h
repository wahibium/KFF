
#ifndef DEBUGOUTPUT
#define DEBUGOUTPUT


#include <stdio.h>
#include <stdarg.h>
#include "cutil_temp.h"

__constant__ float *DEBUG_OUTPUT_C;
float* debug_init_memory(void);
float debug_print(float* DEBUG_OUTPUT_D);
void debug_reset(float* DEBUG_OUTPUT_D);
void PrintLogFile(const char *msg, ...);
#endif

