// ################################################################################
//
// name:          stack.cc
//
// author:        Mohamed Wahib
//
// purpose:       the definition of a class IntStack (a stack for int)
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>
#include <stdlib.h>

#include "stack.h"
#include "memalloc.h"

// ================================================================================
//
// name:          IntStack::Intstack
//
// function:      the constructor for the class Intstack; allocates the memory for 
//                a maximal number of integers to store
//
// parameters:    max..........a maximal number of integers to store
//
// returns:       (none)
//
// last modified: February 1999
//
// ================================================================================

IntStack::IntStack(int max)
{
  maxSize = max;
  size    = 0;

  s = (int*) Calloc(max,sizeof(int));
}

// ================================================================================
//
// name:          IntStack::~Intstack
//
// function:      the destructor for the class Intstack; frees the memory allocted 
//                for the stored integers
//
// parameters:    (none)
//
// returns:       (none)
//
// last modified: February 1999
//
// ================================================================================

IntStack::~IntStack()
{
  Free(s);
}

// ================================================================================
//
// name:          IntStack::push
//
// function:      stores an integer on the top of the stack, crashes when the stack
//                is full (with exit code -1)
//
// parameters:    x............a number to store
//
// returns:       (int) the stored number
//
// last modified: February 1999
//
// ================================================================================

int IntStack::push(int x)
{
  if (size>=maxSize)
    {
      fprintf(stderr,"ERROR: push method called for a full stack!\n");
      exit(-1);
    }

  return s[size++]=x;
}

// ================================================================================
//
// name:          IntStack::pop
//
// function:      pops the integer on the top of the stack and gets rid of it; 
//                crashes when the stack is empty (with exit code -1)
//
// parameters:    (none)
//
// returns:       (int) the number from the top of the stack
//
// last modified: February 1999
//
// ================================================================================

int IntStack::pop()
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

// ================================================================================
//
// name:          IntStack::empty
//
// function:      checks whether the stack is empty
//
// parameters:    (none)
//
// returns:       (int) true if the stack is empty
//
// last modified: February 1999
//
// ================================================================================

int IntStack::empty()
{
  return (size==0);
}

// ================================================================================
//
// name:          IntStack::notEmpty
//
// function:      checks whether the stack is not empty
//
// parameters:    (none)
//
// returns:       (int) true if the stack is not empty
//
// last modified: February 1999
//
// ================================================================================

int IntStack::notEmpty()
{
  return size;
}

// ================================================================================
//
// name:          IntStack::full
//
// function:      checks whether the stack is full
//
// parameters:    (none)
//
// returns:       (int) true if the stack is full
//
// last modified: February 1999
//
// ================================================================================

int IntStack::full()
{
  return (size==maxSize);
}

// ================================================================================
//
// name:          IntStack::getSize
//
// function:      checks the size of the stack
//
// parameters:    (none)
//
// returns:       (int) the current size of the stack
//
// last modified: February 1999
//
// ================================================================================

int IntStack::getSize()
{
  return size;
}
