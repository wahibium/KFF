#ifndef _DATAARRAY_H_
#define _DATAARRAY_H_

#define ArrayIDoffset 1000

struct DataArray
{
	/* data */
	int arrayID;	// Array IDs range from 1001~2000
	char * arrayName;
	int dimension;
	int xsize;
	int ysize;
	int zsize;	
};

#endif