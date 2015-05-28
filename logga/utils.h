#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <string.h>

int SwapInt(int *a, int *b);
int SwapLong(long *a, long *b);
void SwapPointers(void **a, void **b);
int PrintSpaces(FILE *out, int num);

/* Exit function to handle fatal errors*/
inline void err_exit()
{
    printf("[Fatal Error]: %s \nExiting...\n", errorMsg);
    exit(1);
}

#endif
