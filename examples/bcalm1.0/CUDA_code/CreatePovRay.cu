/* 
 * File:   CreatePovRay.cu
 * Author: pwahl
 *
 * Created on December 18, 2009, 2:29 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include "grid.cu"
#include "povray_help.cu"

/*
 * 
 */
int main(int argc, char** argv) {

    my_grid g;
    FILE* file;
    sight_info info;
    printf ("\nInitialize grid\n---\n"); // read in data from the input file
    grid_initialize (&g, argv[2]);
    // Create a POVRAY source  file
    file = fopen("/home/pwahl/Documents/Phd/Simulator/MyCode/PovRay/PovRay/PovBasic.txt","w+"); // set where you want to create the POVRAY file
    
    InitFile(file);   // PovBasic needs to be in the same directory.

    /*for (int k=0;k<10;k++)
    {my_box box;
    float p=(float) k;
    box.xs=p;
    box.ys=0;
    box.zs=0;
    box.xe=p+0.5;
    box.ye=1;
    box.ze=1;

    AddBox(file,box);}*/


    AddAll(file,g,&info);
    SetCamera(file,info);

  
    








    


    return (EXIT_SUCCESS);
}

