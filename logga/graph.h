#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <stdio.h>
#include <cstddef>

char errorMsg[100];
 
/* Adjacency list node*/
struct Adjlist_node
{
    private:   
    int ID;
    struct Adjlist_node *next; /*Pointer to the next node*/

    public:
    Adjlist_node();
    Adjlist_node(int v);
    int getID(){return ID;}
    Adjlist_node * getNext(){return next;}
    void setNext(Adjlist_node * nxtPtr){next=nxtPtr;}

};
 
/* Adjacency list */
struct Adjlist
{
    int num_members;           /*number of members in the list (for future use)*/
    Adjlist_node *head;      /*head of the adjacency linked list*/
};
 
/* Graph structure. A graph is an array of adjacency lists.
   Size of array will be number of vertices in graph*/
class Graph
{
    private:
    int numVertices;         /*Number of vertices*/
    Adjlist *adjListArr;     /*Adjacency lists' array*/
    public:
    Adjlist_node* addNode(int v);
    Graph(int n);
    ~Graph();
    void addEdge(int src, int dest);
    void displayGraph();

};
 

#endif