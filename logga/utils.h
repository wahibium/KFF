#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <string.h>

int swapInt(int *a, int *b);
int swapLong(long *a, long *b);
void swapPointers(void **a, void **b);
int printSpaces(FILE *out, int num);

/* Exit function to handle fatal errors*/
inline void err_exit()
{
    printf("[Fatal Error]: %s \nExiting...\n", errorMsg);
    exit(1);
}

#endif
