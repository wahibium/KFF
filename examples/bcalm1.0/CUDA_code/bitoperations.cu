#ifndef BITOPERATIONS_H
#define BITOPERATIONS_H
void printbitssimple(unsigned long n) {
    unsigned long long i;
    i = (unsigned long long) 1 << 31;
    int cnt = -1;
    while (i > 0) {
        cnt = cnt + 1;
        if (cnt == 8) {
            printf(" ");
            cnt = 0;
        }
        if (n & i)
            printf("1");
        else
            printf("0");
        i >>= 1;
    }
    printf("\n");
}




unsigned long long SetProperty(unsigned long long property, int startbit, int stopbit, int value)
// This function takes in an existing property field and puts value in proterty starting at startbit to stopbit
 {
    unsigned long long selector = ((((unsigned long long) 1) << (stopbit - startbit)) - 1) << startbit; // Take the inverse
    //printbitssimple(selector);
    unsigned long long toinsert = ((unsigned long long) value << startbit);
    //printbitssimple(toinsert);
    return (selector & toinsert) | (property & (~selector));


}



unsigned long long SetPropertyFast(unsigned long property, unsigned long selector, int startbit, int value)
// This function takes in an existing property field and puts value in proterty starting at startbit to stopbit
 {
    unsigned long long toinsert = ((unsigned long long) value << startbit);
    //printbitssimple(toinsert);
    return (selector & toinsert) | (property & (~selector));


}




__device__ unsigned long GetPropertyFast(unsigned long property, unsigned long selector, int startbit)
 {
    return ((property & selector) >> startbit);
}




unsigned long GetPropertyFastHost(unsigned long property, unsigned long selector, int startbit)
 {
    return ((property & selector) >> startbit);
}

unsigned long SetMatType(unsigned long property, int f, int value) // Sets the Mattype of field f to value value. Checks for an existing state.
{
    // get if there is anything before in case we have to addapt.
    unsigned long myprop = GetPropertyFastHost(property, SELECTOR_MATTYPE << f*BIT_MATTYPE_FIELD, f * BIT_MATTYPE_FIELD);
 

    return SetPropertyFast(property, SELECTOR_MATTYPE << f*BIT_MATTYPE_FIELD, f*BIT_MATTYPE_FIELD, value|myprop);

}


unsigned long SetAmatType(unsigned long property, int f, int value) // Sets the Mattype of field f to value value. Checks for an existing state.
{
    // get if there is anything before in case we have to addapt.

    return SetPropertyFast(property, SELECTOR_AMAT << f*BIT_AMAT_FIELD, f*BIT_AMAT_FIELD, value);

}


unsigned long SetCpmlIndex(unsigned long property, int f, int value) // Sets the CPMLINDEX of field f to value value. Checks for an existing state.
 {
    // get if there is anything before in case we have to addapt.
    unsigned long myprop = GetPropertyFastHost(property, SELECTOR_CPMLINDEX << f*BIT_CPMLINDEX_FIELD, f * BIT_CPMLINDEX_FIELD);

    switch (myprop) {
           
  
        default: // no problem
            return SetPropertyFast(property, SELECTOR_CPMLINDEX << f*BIT_CPMLINDEX_FIELD, f*BIT_CPMLINDEX_FIELD, value);

    }
}

#endif // MY_BITOPERATIONS_H
