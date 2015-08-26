
#ifndef MY_LORENTZ_HELP_H
#define MY_LORENTZ_HELP_H
#include <stdio.h>
#include <unistd.h>
#include "grid.cu"
#include "DebugOutput.h"

//#include <cutil.h>
#include "cutil_temp.h"// This file was around. I used it because nvcc could not find functions like CUDA_SAFE_CALL, presumebly present in cutil.h
#include "ConstantMemoryInit.h"
#include "bitoperations.cu"
#include "ArrayHandling.cu"

void populateconstantlorentz(my_grid* g) // Creates the array with constants that can be stored in constant memory
{
    float alphap, xip, gammam, gammasum, wm, sigma, dt, eps, wpm, tempgamma; // help variables to calculate the coefficients.
    int np; // number of poles.
    dt = g->dt;
    //Initialize the amount of lorentz poles to zero.
    g->max_lorentz_poles = 0;


    for (int mat = 0; mat < g->ll; mat++) {
        gammasum = 0; //Init before we calculate the sum
        np = g->lorentz[mat * (NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ) + POS_NP_LORENTZ];
        sigma = g->lorentz[mat * (NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ) + POS_SIGMA_LORENTZ];
        eps = g->lorentz[mat * (NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ) + POS_EPS_LORENTZ];

        // Find the maximal amount of poles in the defined materials.

        if (g->max_lorentz_poles < np) {
            g->max_lorentz_poles = np;
            printf("The maximal amount of poles is %d ", g->max_lorentz_poles);
        }

        if (g->max_lorentz_poles > N_POLES_MAX) {
            printf("\n Error: A lorentz material has too many poles defined. BCALM is compiled to allow for maximally %d poles.\n", N_POLES_MAX);
            printf("Change the macro N_POLES_MAX in ConstantMemoryInit.h to the wanted value and rescompile");
            assert(0);
        }




        // printf("\n sigma %E \n",sigma);
        //printf("\n eps %E \n",eps);


        //Eps_r(omega)=1+sum_poles_m{(wpm^2)/(wm^2-omega^2+iwgammam)} // As is Ekins document for with the dispersion parameters of gold.


        for (int pole = 0; pole < np; pole++)// sum over all the poles
        {
            wm = gld(mat, pole, POS_OMEGAM_LORENTZ);
            gammam = gld(mat, pole, POS_GAMMAM_LORENTZ);
            wpm = gld(mat, pole, POS_OMEGAPM_LORENTZ);
            //printf("\nwm %E \n",wm);
            //printf("\ngammam %E \n",gammam);
            //printf("\nwpm %E \n",wpm);
            alphap = (2 - pow(wm*dt, 2)) / (1 + gammam * dt / 2); //recalculated from (9.49a.) in Taflove
            xip = (gammam / 2 * dt - 1) / (gammam / 2 * dt + 1); //recalculated from (9.49a.) in Taflove
            tempgamma = EPSILON_0 * (pow((wpm * dt), 2)) / (1 + gammam / 2 * dt); //recalculated from (9.49a.) in Taflove
            gammasum = gammasum + tempgamma;
            // printf("\naplhap %e \n",xip);
            //printf("\nxip %e \n",xip);
            //printf("\ntempgamma %e \n",tempgamma);

            glc(mat, pole, POS_ALPHAP_LORENTZ_C) = alphap;
            glc(mat, pole, POS_XIP_LORENTZ_C) = xip;
            glc(mat, pole, POS_GAMMAP_LORENTZ_C) = tempgamma;
        }
        //printf("\ngamma= %E \n",gammap);
        float C1 = (0.5 * gammasum) / (2 * EPSILON_0 * eps + 0.5 * gammasum + sigma * dt);
        float C2 = (2 * eps * EPSILON_0 - sigma * dt) / (2 * EPSILON_0 * eps + 0.5 * gammasum + sigma * dt);
        float C3 = (2 * dt) / (2 * EPSILON_0 * eps + 0.5 * gammasum + sigma * dt);
        //printf("C3=%e\n",C3);

        g->lorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C1_LORENTZ_C] = C1;
        g->lorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C2_LORENTZ_C] = C2;
        g->lorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C3_LORENTZ_C] = C3;
        g->lorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_NP_LORENTZ_C] = np;
    }
    /*for (int cnt=0; cnt<g->ll; cnt++)
    {     printf("\n-----------\n");
    for (int cnt2=0; cnt2<(NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C);cnt2++)
    {
   printf(" %e\n ",g->lorentzreduced[cnt*(NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C)+cnt2]);
    }
}*/

}

void populatelorentz(my_grid* g) // Populates the mattypes with the information of being a lorentzcell populates the property array with the address to the right material
{
    int index; // Variables to help fill the array
    int x; // Variables to help fill the array
    int y; // Variables to help fill the array
    int z; // Variables to help fill the array
    int dx; // Variables to help fill the array
    int dy; // Variables to help fill the array
    int dz; // Variables to help fill the array
    for (int zone = 0; zone < g->zlzl; zone++) {
        index = g->lorentzzone[zone * NPLORENTZZONE + 0];
        x = g->lorentzzone[zone * NPLORENTZZONE + 1];
        y = g->lorentzzone[zone * NPLORENTZZONE + 2];
        z = g->lorentzzone[zone * NPLORENTZZONE + 3];
        dx = g->lorentzzone[zone * NPLORENTZZONE + 4];
        dy = g->lorentzzone[zone * NPLORENTZZONE + 5];
        dz = g->lorentzzone[zone * NPLORENTZZONE + 6];

        for (int i = x; i <= x + dx; i++) {
            for (int j = y; j <= y + dy; j++) {
                for (int k = z; k <= z + dz; k++) {

                    g->property[PROPNUM_LORENTZ][(i) + (j) * g->xx + (k) * g->xx * g->yy] = SetPropertyFast(g->property[PROPNUM_LORENTZ][(i) + (j) * g->xx + (k) * g->xx * g->yy], SELECTOR_LORENTZ, FIRST_BIT_LORENTZ, index);
                    //printf("\n x=%d y=%d z=%d \n",i,j,k);
                    //printbitssimple (g->property[PROPNUM_LORENTZ][(i) + (j)*g->xx+ (k)*g->xx*g->yy]);
                    for (int f = 0; f < 3; f++) // Only for the electrical field.
                    {
                        // Set the mattype to lorentz in the three directions
                        g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy] = \
                        SetMatType(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, LORENTZ);
                        // Set the index of lorentzmaterial to index in the three directions
                        g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy] =  \
                        SetAmatType(g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, index);

                    }

                }
            }
        }

    }

}

void populatelorentzfield(my_grid *g, my_grid *gd)// makes an List of all the lorentzfields present in the simulation.
//Each cell contains the position of the cell the index of the lorentz material The electric field and the Jp's for each direction and each pole.
// A seperate kernel will use this so that all the processor cores will process lorentzcells simulataneously.
// Also the last cell of a lorentzcube in each direction
{
    int dx; // Variables to help fill the array
    int dy; // Variables to help fill the array
    int dz; // Variables to help fill the array
    int ncells = 0; // Position of each cell in the array
    int nfields = 0; // Number of fields


    // Counting how many cells are out there.
    for (int zone = 0; zone < g->zlzl; zone++) {
        dx = g->lorentzzone[zone * NPLORENTZZONE + 4];
        dy = g->lorentzzone[zone * NPLORENTZZONE + 5];
        dz = g->lorentzzone[zone * NPLORENTZZONE + 6];

        ncells = ncells + (dx + 1)*(dy + 1)*(dz + 1);

    }
    nfields = ncells * 3; // Assuming each lorentz cell is needed
    if (nfields % B_XX_LORENTZ != 0)
        nfields = nfields + (B_XX_LORENTZ - (nfields % B_XX_LORENTZ)); //Pad to fit the threadblock size.

    g->nlorentzfields = nfields;


    // allocate memory for an array of pointers. On pointer for each cell.

    g->lorentzfield = (float*) cust_alloc(sizeof (float) * nfields * NP_TOT_LORENTZ_CL);
    g->lorentzfieldaddress = (float**) cust_alloc(sizeof (float*) * nfields);
    int cnt = 0;
    for (int f = 0; f < 3; f++) {
        for (int k = 0; k < g->zz; k++) {
            for (int j = 0; j < g->yy; j++) {
                for (int i = 0; i < g->xx; i++) {
                    unsigned long proplorentz = g->property[PROPNUM_LORENTZ][(i) + (j) * g->xx + (k) * g->xx * g->yy];
                    int toggle = 0;
                    unsigned long matparam = g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy];

                    if (GetPropertyFastHost(matparam, SELECTOR_MATTYPE << f * BIT_MATTYPE_FIELD, f * BIT_MATTYPE_FIELD) & LORENTZ)
                        toggle = 1;


                    if (toggle == 1) { // Create a Lorentzcell.

                        g->lorentzfieldaddress[cnt] = &gd->field[f][((i) + (j) * g->xx + (k) * g->xx * g->yy)]; // Address to get and store the fields.
                        glcl(cnt, 0, 0, POS_INDEX_LORENTZ_CL) = GetPropertyFastHost(proplorentz, SELECTOR_LORENTZ, FIRST_BIT_LORENTZ); //Get the material.
                        cnt = cnt + 1;


                    }
                }
            }
        }
    }
    // Do the padding.
    for (int pad = cnt; pad < g->nlorentzfields; pad++) {
        g->lorentzfieldaddress[pad] = NULL;
    }






}




















#endif
