// ################################################################################
//
// name:          graph.cc      
//
// author:        Mohamed Wahib
//
// purpose:       the definition of classes Adjlist_node, Adjlist and Graph
//                for manipulation with graphs
//
// last modified: Feb 2014
//
// ################################################################################
#include <string.h>
#include <stdlib.h>
#include "graph.h"
#include "utils.h"

 
/* Function to create an adjacency list node*/
Adjlist_node::Adjlist_node()
{

    this->next = NULL;

}
Adjlist_node::Adjlist_node(int v)
{
    this->ID = v;
    this->next = NULL;

}

/*  Fuction to add a node*/
Adjlist_node* Graph::addNode(int v)
{
    Adjlist_node *newNode = new Adjlist_node(v);
    if(!newNode){ 
        strcpy(errorMsg,"Unable to allocate memory for new node");
        err_exit();
    }
    return newNode;

}
 
/* Constructor to a graph with n vertices*/
Graph::Graph(int n)
{

    this->numVertices = n;
     
    /* Create an array of adjacency lists*/
    this->adjListArr = new Adjlist[n];

    if(!this->adjListArr)
    {
        strcpy(errorMsg,"Unable to allocate memory for adjacency list array");
        err_exit();
    }
    for(int i = 0; i < n; i++)
    {
        this->adjListArr[i].head = NULL;
        this->adjListArr[i].num_members = 0;
    }
 
}
 
/*Destroys the graph*/
Graph::~Graph()
{
    Adjlist_node *tmp; 
    Adjlist_node *adjListPtr = new Adjlist_node;
    /*Free up the nodes*/
    for (int v = 0; v < this->numVertices; v++)
    {
        adjListPtr = this->adjListArr[v].head;
        while (adjListPtr)
        {
            tmp = adjListPtr;
            adjListPtr = adjListPtr->getNext();
            delete tmp;
        }
    }
    /*Free the adjacency list array*/
    delete this->adjListArr;
        
        
}
 
/* Adds an edge to a graph*/
void Graph::addEdge(int src, int dest)
{
    /* Add an edge from src to dst in the adjacency list*/
    Adjlist_node *newNode = addNode(dest);
    newNode->setNext(this->adjListArr[src].head);
    this->adjListArr[src].head = newNode;
    this->adjListArr[src].num_members++;
 
}
 
/* Function to print the adjacency list of graph*/
void Graph::displayGraph()
{
    Adjlist_node *adjListPtr = new Adjlist_node;
    for (int i = 0; i < this->numVertices; i++)
    {
        adjListPtr = this->AdjListArr[i].head;
        printf("\n%d: ", i);
        while (adjListPtr)
        {
            printf("%d->", adjListPtr->getID());
            adjListPtr = adjListPtr->getNext();
        }
        printf("NULL\n");
    }
}

//  Sample of using graph
/*
    Graph *orderExecGraph = new Graph(n);
    if(!orderExecGraph)
    {    
        strcpy(errorMsg,"Unable to allocate memory for order-of-execution graph");
        err_exit();
    }
    
    addEdge(undir_graph, 0, 1);
    addEdge(undir_graph, 0, 4);
    addEdge(undir_graph, 1, 2);
    addEdge(undir_graph, 1, 3);
    addEdge(undir_graph, 1, 4);
    addEdge(undir_graph, 2, 3);
    addEdge(undir_graph, 3, 4);
 
    addEdge(dir_graph, 0, 1);
    addEdge(dir_graph, 0, 4);
    addEdge(dir_graph, 1, 2);
    addEdge(dir_graph, 1, 3);
    addEdge(dir_graph, 1, 4);
    addEdge(dir_graph, 2, 3);
    addEdge(dir_graph, 3, 4);
 
    printf("\nUNDIRECTED GRAPH");
    displayGraph(undir_graph);
    destroyGraph(undir_graph);
 
    printf("\nDIRECTED GRAPH");
    displayGraph(dir_graph);
    destroyGraph(dir_graph);
 

*/