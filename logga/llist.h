#ifndef _LLIST_H_
#define _LLIST_H_

struct Node {
  int ID;
  node *next;
};

void addNode(Node **list, int ID);

int removeNode(Node **list, int ID);

#endif
