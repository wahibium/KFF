#ifndef PRINTLOG
#define PRINTLOG

#include <stdio.h>
#include <stdarg.h>

extern FILE * simout;
extern char outdir[512];
#define SIMOUTFILE "simout.txt"
#define DEBUGOUTFILE "debugout.txt"

void PrintLog(const char *msg, ...);

#endif

