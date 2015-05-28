#ifndef _STACK_H_
#define _STACK_H_

class IntStack {

 private:

  int maxSize;
  int size;
  int *s;

 public:

  IntStack(int max);
  ~IntStack();

  int Push(int x);
  int Pop();

  int Empty();
  int NotEmpty();
  int Full();
  int GetSize();
};


#endif
