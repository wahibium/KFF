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

  int push(int x);
  int pop();

  int empty();
  int notEmpty();
  int full();
  int getSize();
};


#endif
