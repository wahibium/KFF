#include <stdio.h>
#include <stdlib.h>

#include "stack.h"
#include "memalloc.h"

IntStack::IntStack(int max)
{
  maxSize = max;
  size    = 0;

  s = (int*) Calloc(max,sizeof(int));
}


IntStack::~IntStack()
{
  Free(s);
}

int IntStack::Push(int x)
{
  if (size>=maxSize)
    {
      fprintf(stderr,"ERROR: push method called for a full stack!\n");
      exit(-1);
    }

  return s[size++]=x;
}

int IntStack::Pop()
{
  if (size>0)
    return s[--size];
  else
    {
      fprintf(stderr,"ERROR: pop method called for an empty stack!\n");
      exit(-1);
    }

  return 0;
}

int IntStack::Empty()
{
  return (size==0);
}

int IntStack::NotEmpty()
{
  return size;
}

int IntStack::Full()
{
  return (size==maxSize);
}

int IntStack::GetSize()
{
  return size;
}
