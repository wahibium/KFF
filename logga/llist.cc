#include "llist.h"

void AddNode(Node **list, int ID)
{
	Node *p = new Node;
	p->ID   = ID;
	p->next = *list;
	list    = p;
}

int RemoveNode(Node **list, int ID)
{
	Node *current = *list;
	Node *previous = *list;
	if (*list == NULL)
		return -1;
	while (current->next != NULL)
	{
		if (current->ID == ID)
		{
			previous->next = current->next;
			*list = previous;
			delete current;

			return 1;
		}	
		previous = current;
		current = current->next;

	}

	return 0;
}