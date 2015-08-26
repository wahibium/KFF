//rantest.spg  processed by SPAG 4.52O  at 11:03 on 18 Oct 1996
#include "header.h"
void RANTEST(int Iseed){
    //     ---test and initialize the random number generator
    int i;
    
    RANSET(Iseed);
    cout << "********** test random numbers **********" << endl;
    for(i=1;i<=5;i++){
	cout << "i, ranfrk() " << i << " " << RANFRK() << endl;
    }
}//  END SUBROUTINE RANTEST
  
