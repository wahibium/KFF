#ifndef TOOLS_H
#define TOOLS_H

#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <string.h>
#include <math.h>






// allocate memory
void * cust_alloc (size_t size) {
	register void *value = malloc (size); // allocate a block of memory
	if (value == NULL) // make sure we suceeded in allocating the desired memory
		error (-1, 0, "Virtual memory exhausted.");
	else
		memset (value, 0, size); 
	return value;
	/*This could be replaced with a #define statement:
	 * #define CALLOC(r,s,t) if(((r)=calloc(s,t)) == NULL){error(-1, 0, "Virtual memory exhausted.");} //r is pointer to array, s is the size of the array, t is the byte size of an array element
	 * this might be slightly faster and have less overhead, or totally pointless :)
	 */
}



#endif
