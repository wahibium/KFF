#include <stdio.h>

#include "utils.h"

int SwapInt(int *a, int *b)
{
  int aux;
  
  aux = *a;
  *a  = *b;
  *b  = aux;

  return 0;
}

int SwapLong(long *a, long *b)
{
  long aux;
  
  aux = *a;
  *a  = *b;
  *b  = aux;

  return 0;
}

void SwapPointers(void **a, void **b)
{
  void *aux;

  aux = *a;
  *a  = *b;
  *b  = aux;
}


int PrintSpaces(FILE *out, int num)
{
  int i;
 
  for (i=0; i<num; i++)
    fputc(' ',out);

  return 0;
}
