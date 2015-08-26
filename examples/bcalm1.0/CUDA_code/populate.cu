#ifndef POPULATE_H
#define POPULATE_H
#include <stdio.h>
#include "grid.cu"
//#include <cutil.h>
#include "cutil_temp.h"// This file was around. I used it because nvcc could not find functions like CUDA_SAFE_CALL, presumebly present in cutil.h
#include "ConstantMemoryInit.h"
#include "cpml_help.cu"
#include "lorentz_help.cu"
#include "bitoperations.cu"
#include "region_distance_help.cu"

/* Print n as a binary number */


void populateepsilon(my_grid* g) {
    // Population of the epsilonzones assuming that the grid is allready initialized on zeros everywhere.
    //Zeros will point to the first dielectric in constant memory assumed to be the ambient dielectric
    int index; // Variables to help fill the array
    region R; // Store the region parameters
    for (int zone = 0; zone < g->zdzd; zone++) {
        index = g->dielzone[zone * NPDIELZONE + 0];
        R.x = g->dielzone[zone * NPDIELZONE + 1];
        R.y = g->dielzone[zone * NPDIELZONE + 2];
        R.z = g->dielzone[zone * NPDIELZONE + 3];
        R.dx = g->dielzone[zone * NPDIELZONE + 4];
        R.dy = g->dielzone[zone * NPDIELZONE + 5];
        R.dz = g->dielzone[zone * NPDIELZONE + 6];
        R.type = g->dielzone[zone * NPDIELZONE + POS_TYPE_DEPS];
        //printf("type= %d",R.type);

        for (int i = R.x; i <= R.x + R.dx; i++) {
            for (int j = R.y; j <= R.y + R.dy; j++) {
                for (int k = R.z; k <= R.z + R.dz; k++) {
                    if (InZone(g, R, i, j, k) == 1)
                        //Isotropic
                        // PROPNUM_DEPS is populated, but deprecated.
                        // PROPNUM_DEPS is not used in the kernel,
                        // instead, PROPNUM_AMAT is used.
                        g->property[PROPNUM_DEPS][(i) + (j) * g->xx + (k) * g->xx * g->yy] =\
                        SetProperty(g->property[PROPNUM_DEPS][(i) + (j) * g->xx + (k) * g->xx * g->yy], FIRST_BIT_DEPS, FIRST_BIT_DEPS + BIT_DEPS - 1, index);
                    for (int f=0;f<3;f++)
                    {// Anisotropic
                    g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy] = \
                    SetAmatType(g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy], f,index);
                    }


                }
            }
        }

    }



}

// Populates the the property array for sources.

void populatesource(my_grid *g) {
    // Uses BIT_SOURCE bits in the property array to indicate the address of the source and weather the cell is a source or not.

    int x; // Variables to help fill the array
    int y; // Variables to help fill the array
    int z; // Variables to help fill the array

    if (g->ss >= (int) pow((float) 2, (float) BIT_SOURCE)) {
        printf("Error:There are too many sources defined in this simulation");
        assert(0);
    }
    for (int source = 0; source < g->ss; source++) {
        x = (int) g->source[source * NPSOURCE];
        y = (int) g->source[source * NPSOURCE + 1];
        z = (int) g->source[source * NPSOURCE + 2];


        //printf("\nx=%d,y=%d,z=%d\n",x,y,z);
        g->property[PROPNUM_SOURCE][(x) + (y) * g->xx + (z) * g->xx * g->yy] = SetProperty(g->property[PROPNUM_SOURCE][(x) + (y) * g->xx + (z) * g->xx * g->yy], FIRST_BIT_SOURCE, FIRST_BIT_SOURCE + BIT_SOURCE, source);
        //printbitssimple(g->property[PROPNUM_SOURCE][(x) + (y)*g->xx+ (z)*g->xx*g->yy]);
        for (int f = 0; f < 6; f++) {
            if (pow(g->source[source * NPSOURCE + POS_ABS_SOURCE + f], 2) > 0) {
                g->property[PROPNUM_MATTYPE][(x) + (y) * g->xx + (z) * g->xx * g->yy] = SetMatType(g->property[PROPNUM_MATTYPE][(x) + (y) * g->xx + (z) * g->xx * g->yy], f, SOURCE);
                //printbitssimple(g->property[PROPNUM_MATTYPE][(x) + (y)*g->xx+ (z)*g->xx*g->yy]);
                //printf("\nField is %d\n",f);
            }
        }

    }

}

void populateperfectlayerzone(my_grid *g) {
    // Uses BIT_PERFECTLAYER bits in the property array to indicate which fields have to be set to zero

    int value, x, y, z, dx, dy, dz; // Variables to help fill the array


    for (int zone = 0; zone < g->pp; zone++) {
        value = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 0];
        x = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 1];
        y = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 2];
        z = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 3];
        dx = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 4];
        dy = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 5];
        dz = (int) g->perfectlayerzone[zone * NPPERFECTLAYER + 6];

        for (int i = x; i <= x + dx; i++) {
            for (int j = y; j <= y + dy; j++) {
                for (int k = z; k <= z + dz; k++) {


                    for (int f = 0; f < 6; f++) {
                        if (value & (1 << f)) {
                            //printf("\nperfect layer at for f=%d\n",f);
                            g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy] = SetMatType(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, PERFECTLAYER);
                        }
                    }
                    //printbitssimple(g->property[PROPNUM_MATTYPE][(i) + (j)*g->xx+ (k)*g->xx*g->yy]);

                }
            }
        }

    }

}

void correctborders(my_grid *g) // This function goes trough the whole grid and corrects for bordercases.
{
    int next [3][3] = {
        {1, 0, 0}, //x
        {0, 1, 0}, //y
        {0, 0, 1} //z
    };
    int index;


    for (int f = 0; f < 3; f++) {
        for (int k = 0; k < g->zz; k++) {
            for (int j = 0; j < g->yy; j++) {
                for (int i = 0; i < g->xx; i++) {
                    if (((i + next[f][0]) < g->xx) && ((j + next[f][1]) < g->yy) && ((k + next[f][2]) < g->zz)) {
                        // Get the material property
                        unsigned long matparam = g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy]; // material type
                        int matn = GetPropertyFastHost(matparam, SELECTOR_MATTYPE << f*BIT_MATTYPE_FIELD, f * BIT_MATTYPE_FIELD);
                        // Get the material next in the field direction
                        unsigned long matparamp1 = g->property[PROPNUM_MATTYPE][(i + next[f][0]) + (j + next[f][1]) * g->xx + (k + next[f][2]) * g->xx * g->yy]; // material type
                        int matnp1 = GetPropertyFastHost(matparamp1, SELECTOR_MATTYPE << f*BIT_MATTYPE_FIELD, f * BIT_MATTYPE_FIELD);

                        // Here we check if the next cell is also a lorentz cell.
                        // If not we are on an edge. Therfore make the following changes(example in X)
                        // if Ex(xyz) = LORENTZ and Ex(x+1yz) != LORENTZ---> Ex(xyz)=DIEL of cell (x+1yz)
                        if (matn & LORENTZ) {
                            if ((~matnp1 & LORENTZ)) {
                                // Ex(xyz becomes dielectric(everybody is) by resetting the lorentz bit)
                                //printf("Before\n");
                                //printbitssimple(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy]);
                                g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy] =     \
                                    SetPropertyFast(matparam, LORENTZ << f*BIT_MATTYPE_FIELD, f*BIT_MATTYPE_FIELD, 0);
                                //printf("After\n");
                               // printbitssimple(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy]);
                                //printf("mnp1\n");
                                //printbitssimple((~matnp1 & LORENTZ));
                                // Get the dielectric contstant of the next cell.
                                index = GetPropertyFastHost(g->property[PROPNUM_DEPS][(i + next[f][0]) + (j + next[f][1]) * g->xx + (k + next[f][2]) * g->xx * g->yy], SELECTOR_DEPS, FIRST_BIT_DEPS);
                                // Copy it to the actual cell.
                                g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy] =  \
                                SetAmatType(g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, index);



                            }
                        }
                    }
                }
            }
        }
    }


}

void populategrid(my_grid* gh, my_grid* gd) {
    PrintLog("Populate...");
    // Populate the propertyarray for epsilon
    populateepsilon(gh);
    // populate constant lorentz
    populateconstantlorentz(gh);
    // Populate the mattypes+
    populatelorentz(gh);
    // BorederCases
    correctborders(gh);

    //populate the cell array for the grid
    populatelorentzfield(gh, gd);

    // Update the border Cases
    if ((gh->zczc) > 0)
        GetUpdateBorderCpmlZones(gh);
    else
        gh->ncpmlcells = 0;
    //populatecpml is done at cell level because the termination by a PEC or PMC determines wheather some fields are treated as CPML.        // Populate the source
    populatesource(gh);
    // Populate the perfect layers
    populateperfectlayerzone(gh);

    PrintLog("Populate Successfull!\n");

    //Poputlate the property array for lorentz cells
    //Populate the mattype array for lorentz cells

}




/* unsigned long long test = 54564654654;
 int test2;
 printbitssimple(test);
 test=SetProperty(test,5,15,4);
  printbitssimple(test);
 test2=GetProperty(test,5,15);
 printf("test is %d\n",test2);

  printbitssimple(test);
 test=SetProperty(test,35,45,8);
  printbitssimple(test);
 test2=GetProperty(test,35,45);
 printf("test is %d\n",test2);*/

#endif // POPULATE_H
