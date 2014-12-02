#ifndef _MEMALLOC_H_
#define _MEMALLOC_H_

#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>

// inline functions

inline void *Malloc(long x)
{
   void *p;

   p=malloc(x);

   if (p==NULL)
   {
      printf("ERROR: Not enough memory (for a block of size %lu)\n",x);
      exit(-1);
   }

   return p;
}

inline void *Calloc(int x, int s)
{
   void *p;

   p=calloc((int) x, (int) s);

   if (p==NULL)
   {
      printf("ERROR: Not enough memory. (for a block of size %lu)\n",x*s);
      exit(-1);
   }

   return p;
}

inline void Free(void *x)
{
  free(x);
}

#endif
