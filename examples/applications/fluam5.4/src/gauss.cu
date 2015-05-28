#include "header.h"

double gauss (){

//    *******************************************************************
//    ** RANDOM VARIATE FROM THE STANDARD NORMAL DISTRIBUTION.         **
//    **                                                               **
//    ** THE DISTRIBUTION IS GAUSSIAN WITH ZERO MEAN AND UNIT VARIANCE.**
//    **                                                               **
//    ** REFERENCE:                                                    **
//    **                                                               **
//    ** KNUTH D, THE ART OF COMPUTER PROGRAMMING, (2ND EDITION        **
//    **    ADDISON-WESLEY), 1978                                      **
//    **                                                               **
//    ** ROUTINE REFERENCED:                                           **
//    **                                                               **
//    ** REAL FUNCTION RANF ( DUMMY )                                  **
//    **    RETURNS A UNIFORM RANDOM VARIATE ON THE RANGE ZERO TO ONE  **
//    *******************************************************************

    
    const double A1 = 3.949846138;
    const double A3 = 0.252408784;
    const double A5 = 0.076542912;
    const double A7 = 0.008355968;
    const double A9 = 0.029899776;

    double SUM, R, R2, G;
    

    SUM = 0.0;

    //for(I=1;I<=12;I++){
	
    SUM = SUM + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK() + RANFRK();
	//SUM = SUM + XRANDXXX();
	//}

    R  = ( SUM - 6.0 ) * 0.25;
    R2 = R * R;

    G = (((( A9 * R2 + A7 ) * R2 + A5 ) * R2 + A3 ) * R2 +A1 ) * R;

    return G;
}// END FUNCTION GAUSS


/*     FUNCTION RANF ( DUMMY )

!    *******************************************************************
!    ** RETURNS A UNIFORM RANDOM VARIATE IN THE RANGE 0 TO 1.         **
!    **                                                               **
!    **                 ***************                               **
!    **                 **  WARNING  **                               **
!    **                 ***************                               **
!    **                                                               **
!    ** GOOD RANDOM NUMBER GENERATORS ARE MACHINE SPECIFIC.           **
!    ** PLEASE USE THE ONE RECOMMENDED FOR YOUR MACHINE.              **
!    *******************************************************************

        INTEGER ::    L, C, M
        PARAMETER ( L = 1029, C = 221591, M = 1048576 )

        INTEGER ::    SEED
        real ::       DUMMY
        SAVE        SEED
        DATA        SEED / 0 /

!    *******************************************************************

        SEED = MOD ( SEED * L + C, M )
        RANF = DBLE ( SEED ) / M

        RETURN
	END FUNCTION*/
