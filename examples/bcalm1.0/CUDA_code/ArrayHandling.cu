
#ifndef ARRAY_HANDLE
#define ARRAY_HANDLE 
// Functions used to delete zones dynamically out of arrays.




float* AddElementToArray(float* InArray,int SizeIn,float* AddEl,int SizeAdd)


{float *NewArray=(float*) cust_alloc(sizeof (float) * (SizeIn+SizeAdd));

for (int cnt=0;cnt<SizeIn;cnt++)// Copy Old elements
{NewArray[cnt]=InArray[cnt];
 }
for (int cnt=0;cnt<SizeAdd;cnt++)// Add New elements
{NewArray[cnt+SizeIn]=AddEl[cnt];
 }

return NewArray;


}


float* DeleteElementArray(float* InArray,int SizeIn,int RemEl,int SizeRem)


{float *NewArray=(float*) cust_alloc(sizeof (float) * (SizeIn-SizeRem));

for (int cnt=0;cnt<RemEl;cnt++)// Copy Old elements
{NewArray[cnt]=InArray[cnt];
 }
for (int cnt=RemEl+SizeRem;cnt<SizeIn;cnt++)// Add other elements
{NewArray[cnt-SizeRem]=InArray[cnt];
 }

return NewArray;


}

void DeleteZone(my_grid *g,int zone)

{
g->cpmlzone=DeleteElementArray(g->cpmlzone,g->zczc*NP_CPML,NP_CPML*zone,NP_CPML);
g->zczc=g->zczc-1;
}


#endif // ARRAY_HANDLE