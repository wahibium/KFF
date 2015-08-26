//**==ranfrk.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
#include "header.h"
#include "header_random.h"
double RANFRK(){
    //
    //     random number generator
    //
    //     Idum (input): can be used as seed (not used in present
    //                  random number generator.
    
    //int Idum;
    //double RANFRK_value;
    //RANFRK_value = RCARRY();
    //return RANFRK_value;
  return RCARRY();
  // ----------------------------------------------------
}//END FUNCTION RANFRK
 
//**==randx.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
  
double XRANDXXX(){
    //----------------------------------------------------------------------
    //  Random number generator, fast and rough, machine independent.
    //  Returns an uniformly distributed deviate in the 0 to 1 interval.
    //  This random number generator is portable, machine-independent and
    //  reproducible, for any machine with at least 32 bits / real number.
    //  REF: Press, Flannery, Teukolsky, Vetterling, Numerical Recipes (1986)
    //----------------------------------------------------------------------
//#include "header_random.h"
    int IA, IC, M1;
    double RM, RANDX_value;
    M1=714025; IA=1366; IC=150889; RM=1.0/M1;
    //cout << "Iseed " << ISEED << endl;
    ISEED = (IA*ISEED+IC) % M1;
    RANDX_value = ISEED*RM;
    //cout << "RANDX_value " << ISEED << " " << RANDX_value << endl;
    if (RANDX_value<0.){
	cout << "*** Random number is negative ***" << endl;
	return 0;
    }

    return RANDX_value;
}//END FUNCTION RANDX
//**==ranset.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
void RANSET(int Iseed){
    //--- initializes random number generator
    //int Iseed;
    RSTART(Iseed);
    return;
}//END SUBROUTINE RANSET
//**==rstart.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
void RSTART(int Iseeda){
    //----------------------------------------------------------------------C
    //       Initialize Marsaglia list of 24 random numbers.
    //----------------------------------------------------------------------C
//#include "header_random.h"
  //double ran;
    int i;
     
    I24 = 24;
    J24 = 10;
    CARRY = 0.;
    ISEED = Iseeda;
    //
    //       get rid of initial correlations in rand by throwing
    //       away the first 100 random numbers generated.
    //
    for(i=1;i<=100;i++){
      //ran = XRANDXXX();
      XRANDXXX();
      //cout << "RSTART " << ran << endl;
    }
    //
    //       initialize the 24 elements of seed
    //
    
    for(i=1;i<=24;i++){
	SEED[i-1] = XRANDXXX();
    }
    
    return;
}//END SUBROUTINE RSTART
//**==rcarry.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
 
double RCARRY(){
    //----------------------------------------------------------------------C
    //       Random number generator from Marsaglia.
    //----------------------------------------------------------------------C
//#include "header_random.h"
    double RCARRY_value, TWOm24, TWOp24, uni;
    TWOp24=16777216.0; TWOm24=1./TWOp24;
    //
    //       F. James Comp. Phys. Comm. 60, 329  (1990)
    //       algorithm by G. Marsaglia and A. Zaman
    //       base b = 2**24  lags r=24 and s=10
    //
    uni = SEED[I24-1] - SEED[J24-1] - CARRY;
    if (uni<0.){
	uni = uni + 1.;
	CARRY = TWOm24;
    }
    else{
	CARRY = 0.;
    }
    SEED[I24-1] = uni;
    I24 = I24 - 1;
    if(I24==0) I24 = 24;
    J24 = J24 - 1;
    if (J24==0) J24 = 24;
    RCARRY_value = uni;
    
    return RCARRY_value;
}//  END FUNCTION RCARRY
  
