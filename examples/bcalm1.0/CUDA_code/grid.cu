#ifndef MY_GRID_H
#define MY_GRID_H

#include <stdlib.h>
//#include <H5LT.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "tools.h"
#define MY_MAXMATPARAMS 4
#include "ConstantMemoryInit.h"



// structure that holds the field values

typedef struct my_grid {
    float *field[6]; // field values (Ex, Ey, ..., Hz)
    int *mat[6]; // material types corresponding to those fields
    int *type_index; // store the material type for every material index
    int *dielzone; //store the dielectrical zones
    int *lorentzzone; //store the lorentzzones
    int *perfectlayerzone; // store the perfect layer zones
    float *cpmlzone; // Store the cpmlparameters
    float *cpmlcell; // List of  all the cpmlcells . Each holds the position of the cell and the necarry parameters to do a CPML update. Since alot of the CPML calculations are local to one cell. We call them in a seperate kernel
    float *params; // holds the parameters for each material
    float dx, dt; // the constants
    int xx, yy, zz, tt, mm, T, symz, ee, ss, zdzd, pp, ll, mp, zlzl, zczc,zozo, ncpmlcells,nlorentzfields,max_lorentz_poles;
    float *C1; // [1-sigma*dt/2*epsilon]/[1+sigma*dt/2*epsilon values;
    float *C2; // holds the [dt/eps]/[1+sigma*dt/2*epsilon] values;
    float *deps; // holds the epsilon values;
    float *sigma; //holds the sigma values;
    float *lorentz; //holds the parameters for each lorentz material in global memory
    float *lorentzfield; // List of all the lorentzfields present in the simulation.
    //Each element contains the position of the field the index of the lorentz material The electric field and the Jp's for each direction and each pole.
    // A seperate kernel will use this so that all the processor cores will process lorentzfield simulataneously.
    float **lorentzfieldaddress;  //Same as lorentzcell except it stores the address as is it an int (up and backcast gives erros for large numbers)
    float *lorentzreduced; // holds the reduced lorentz parameters and is placed in constant memory

    float *source; // constains the the parameters of different sources.
    unsigned long *property[NUM_PROP_TOT]; // hold the property of each cell to be read every time. Hashing will be performed on this number. The first property is read every time. The second one only in special cases.
    float *gridEx, *gridEy, *gridEz; // holds the dt/dx,y,z in si units for each cell
    float *gridHx, *gridHy, *gridHz; // holds the dt/(mudx,y,z) in si units for each cell will be rescaled to accomodate nonuniform grid.
    float *gridX, *gridY, *gridZ; // holds dx dy dz in Si Units

    // CPMLParameters to be stored in constant memory.
    float kappa[THICK_CPML_MAX * 3]; // holds kappa in each direction at half yee cell intervals. First element is the start of the PML.
    float bcpml[THICK_CPML_MAX * 3]; // idem for b (7.102)
    float ccpml[THICK_CPML_MAX * 3]; // idem for c (7.99)




} my_grid;


// load everything in first, then organize it

void grid_initialize(my_grid *g, char *hdf5_filename) {
    hid_t file_id; // file id for hdf file
    int cnt;
    float temp;//Help variable
       // temporary variables
       printf("Opening hdf5 file %s...", hdf5_filename);
    file_id = H5Fopen(hdf5_filename, H5F_ACC_RDONLY, H5P_DEFAULT); // open the hdf5 file
    if (file_id <= 0)
        error(-1, 0, "error.\n");
    printf("file id: %d\n", file_id);

    // first get dx, xx, yy, zz, and dt, tt
    printf("Reading in info variables...\n");
    H5LTread_dataset(file_id, "/info/xx", H5T_NATIVE_INT, &(g->xx));
    H5LTread_dataset(file_id, "/info/yy", H5T_NATIVE_INT, &(g->yy));
    H5LTread_dataset(file_id, "/info/zz", H5T_NATIVE_INT, &(g->zz));
    H5LTread_dataset(file_id, "/info/dt", H5T_NATIVE_FLOAT, &(g->dt));
    H5LTread_dataset(file_id, "/info/tt", H5T_NATIVE_INT, &(g->tt));
    H5LTread_dataset(file_id, "/info/ee", H5T_NATIVE_INT, &(g->ee));
    H5LTread_dataset(file_id, "/info/ss", H5T_NATIVE_INT, &(g->ss));
    H5LTread_dataset(file_id, "/info/zdzd", H5T_NATIVE_INT, &(g->zdzd));
    H5LTread_dataset(file_id, "/info/pp", H5T_NATIVE_INT, &(g->pp));
    H5LTread_dataset(file_id, "/info/zlzl", H5T_NATIVE_INT, &(g->zlzl));
    H5LTread_dataset(file_id, "/info/ll", H5T_NATIVE_INT, &(g->ll));
    H5LTread_dataset(file_id, "/info/mp", H5T_NATIVE_INT, &(g->mp));
    H5LTread_dataset(file_id, "/info/zczc", H5T_NATIVE_INT, &(g->zczc));
    H5LTread_dataset(file_id, "/info/zozo", H5T_NATIVE_INT, &(g->zozo));

    printf("Info variables: (xx=%d, yy=%d, zz=%d, tt=%d, dx=%e, dt=%e,ee=%d,ss=%d\nzdzd=%d,pp=%d,zlzl=%d,ll=%d,mp=%d,zczc=%d,zozo,=%d) \n", g->xx, g->yy, g->zz, g->tt, g->dx, g->dt, g->ee, g->ss,
            g->zdzd, g->pp, g->zlzl, g->ll, g->mp, g->zczc,g->zozo);

    // Creating the fields

    for (cnt = 0; cnt < 6; cnt++) {

        g->field[cnt] = (float *) cust_alloc(sizeof (float) * g->xx * g->yy * g->zz);

    }

    // Assigning the property array
    for (cnt = 0; cnt < NUM_PROP_TOT; cnt++) {
        g->property[cnt] = (unsigned long*) cust_alloc(sizeof (unsigned long) * g->xx * g->yy * g->zz);
    }


    // Assigning the grid Array.
    g->gridEx = (float*) cust_alloc(sizeof (float) * (g->xx));
    g->gridEy = (float*) cust_alloc(sizeof (float) * (g->yy));
    g->gridEz = (float*) cust_alloc(sizeof (float) * (g->zz));
    g->gridHx = (float*) cust_alloc(sizeof (float) * (g->xx));
    g->gridHy = (float*) cust_alloc(sizeof (float) * (g->yy));
    g->gridHz = (float*) cust_alloc(sizeof (float) * (g->zz));
    g->gridX = (float*) cust_alloc(sizeof (float) * (g->xx));
    g->gridY = (float*) cust_alloc(sizeof (float) * (g->yy));
    g->gridZ = (float*) cust_alloc(sizeof (float) * (g->zz));

    PrintLog("Loading Grids...");
    H5LTread_dataset(file_id, "/in/grid/x", H5T_NATIVE_FLOAT, g->gridX); // transfer from hdf5 file
    H5LTread_dataset(file_id, "/in/grid/y", H5T_NATIVE_FLOAT, g->gridY);
    H5LTread_dataset(file_id, "/in/grid/z", H5T_NATIVE_FLOAT, g->gridZ);

    // Renormalize to dx to dt/dx to avoid devisions in the kernel.

    for (cnt = 0; cnt < g->xx; cnt++) {

        if (cnt<(g->xx-1))
        g->gridHx[cnt] = 2 / (g->gridX[cnt+1]+g->gridX[cnt]); // Renormalize the grid for the variable gridding.
        else
        g->gridHx[cnt] = 1 / (g->gridX[cnt]);

        g->gridEx[cnt] = 1/ g->gridX[cnt];
        // printf(" %f ", g->gridHx[cnt]);
    }
    //printf("\n---------------------\n");
    for (cnt = 0; cnt < g->yy; cnt++) {

        if (cnt<(g->yy-1))
        g->gridHy[cnt] = 2 / (g->gridY[cnt+1]+g->gridY[cnt]); // Renormalize the grid for the variable gridding.
        else
        g->gridHy[cnt] = 1 / (g->gridY[cnt]);

        g->gridEy[cnt] = 1 / g->gridY[cnt];
        //printf(" %f ", g->gridHy[cnt]);
    }
    //printf("\n---------------------\n");
    for (cnt = 0; cnt < g->zz; cnt++) {
        if (cnt>(g->zz-1))
        g->gridHz[cnt] = 2 / (g->gridZ[cnt+1]+g->gridZ[cnt]); // Renormalize the grid for the variable gridding.
        else
        g->gridHz[cnt] = 1 / (g->gridZ[cnt]);
        g->gridEz[cnt] = 1 / g->gridZ[cnt];
        //printf(" %f ", g->gridHz[cnt]);
    }
    //printf("\n---------------------\n");
    PrintLog("Done!\n");



    // Assigning the dielzone array
    PrintLog("Loading Dielectric Zones... ");
    g->dielzone = (int*) cust_alloc(sizeof (int) * g->zdzd * NPDIELZONE);
    if (g->zdzd > 0) {
        H5LTread_dataset(file_id, "/in/dielzone", H5T_NATIVE_INT, g->dielzone); // transfer from hdf5 file
    } else
        PrintLog("\t No Dielectric Zones defined... ");
    PrintLog("Done!\n");
 /*     for (cnt=0; cnt<g->zdzd; cnt++)
    {     printf("\n-----------\n");
    for (int cnt2=0; cnt2<NPDIELZONE;cnt2++)
    {
   printf(" %d ",g->dielzone[cnt*NPDIELZONE+cnt2]);}

    }*/


    PrintLog("Loading Dielectric Materials... ");
    g->deps = (float*) cust_alloc(sizeof (float) * NDEPS);
    g->sigma = (float*) cust_alloc(sizeof (float) * NDEPS);

    g->C1 = (float*) cust_alloc(sizeof (float) * NDEPS);
    g->C2 = (float*) cust_alloc(sizeof (float) * NDEPS);
    if (g->ee > 0) {
        H5LTread_dataset(file_id, "/parameters/deps", H5T_NATIVE_FLOAT, g->deps); // transfer from hdf5 file
        H5LTread_dataset(file_id, "/parameters/sigma", H5T_NATIVE_FLOAT, g->sigma); // transfer from hdf5 file
    } else
        PrintLog("\t No Dielectric Materials defined... ");
    PrintLog("Done!\n");
    // Renormelize epsr to 1/(epsr*eps0)
    for (cnt = 0; cnt < g->ee; cnt++) {
        temp=(g->sigma[cnt]*g->dt)/(2*g->deps[cnt]);

        g->C1[cnt] = (1-temp)/(1+temp);
        g->C2[cnt] = (g->dt / (g->deps[cnt] * EPSILON_0))/(1+temp);
                   
        //printf("\nsigma=%.4eC1=%.4e,C2=%.4e",g->sigma[cnt],g->C1[cnt],g->C2[cnt] );
    }

    //***LORENTZ***

    //  Assinging the lorentzpropertiesarray

    PrintLog("Reading Lorentz Materials ... ");
    g->lorentz = (float*) cust_alloc(sizeof (float) * g->ll * (NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ));
    if (g->ll > 0) {
        H5LTread_dataset(file_id, "/parameters/lorentz", H5T_NATIVE_FLOAT, g->lorentz); // transfer from hdf5 file
        /*for (cnt = 0; cnt < g->ll; cnt++) {
            printf("\n-----------\n");
            for (int cnt2 = 0; cnt2 < NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ; cnt2++) {
                printf(" %f ", g->lorentz[cnt * (NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ) + cnt2]);
            }
        }*/
    } else
        PrintLog("No Lorentz Materials defined... ");
    PrintLog("Done!\n");

    //Allocate memory for the reduced lorentz array(more than needede because is is going to constant memory and dynamic allaocation is not possible)
    g->lorentzreduced = (float*) cust_alloc(sizeof (float) * NLORENTZ * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C));
   
    // Assigning the Lorentzzone array
    PrintLog("Loading the Lorentzzones... ");
    g->lorentzzone = (int*) cust_alloc(sizeof (int) * g->zlzl * NPLORENTZZONE);
    if (g->zlzl > 0)
        H5LTread_dataset(file_id, "/in/lorentzzone", H5T_NATIVE_INT, g->lorentzzone); // transfer from hdf5 file
    else
        PrintLog("No Lorentzzones defined... ");
    PrintLog("Done!\n");




    /*for (cnt=0; cnt<g->zlzl; cnt++)
  {     printf("\n-----------\n");
  for (int cnt2=0; cnt2<NPLORENTZZONE;cnt2++)
  {
 printf(" %d ",g->lorentzzone[cnt*NPLORENTZZONE+cnt2]);}

  }*/

    //***SOURCES***

    PrintLog("Loading Sources ...");
    g->source = (float*) cust_alloc(sizeof (float) * g->ss * NPSOURCE);
    if (g->ss > 0)
        H5LTread_dataset(file_id, "/parameters/source", H5T_NATIVE_FLOAT, g->source); // transfer from hdf5 file
    else
        PrintLog("No Sources defined... ");

    PrintLog("Done!\n");
    /*for (cnt=0; cnt<g->ss; cnt++)
    {     printf("\n-----------\n");
    for (int cnt2=0; cnt2<NPSOURCE;cnt2++)
    {
              printf(" %f ",g->source[cnt*NPSOURCE+cnt2]);}
                    
        }*/

    //***CPML***
    
    // input the perfectlayer zone
    PrintLog("Loading Perfectlayer Zones ...");
    g->perfectlayerzone = (int*) cust_alloc(sizeof (int) * g->pp * NPPERFECTLAYER);
    if (g->pp > 0)
        H5LTread_dataset(file_id, "/in/perfectlayerzone", H5T_NATIVE_INT, g->perfectlayerzone); // transfer from hdf5 file
    else
        PrintLog("No Perfectlayerzones defined... ");
    /*for (cnt=0; cnt<g->pp; cnt++)
        {     printf("\n-----------\n");
        for (int cnt2=0; cnt2<NPPERFECTLAYER;cnt2++)
        {
                  printf(" %d ",g->perfectlayerzone[cnt*NPPERFECTLAYER+cnt2]);}

            }*/
        PrintLog("Done!\n");

    // input the cpmlzones

    PrintLog("Loading CPML Zones... ");
    g->cpmlzone = (float*) cust_alloc(sizeof (float) * g->zczc * NP_CPML);
    if (g->zczc > 0)
        H5LTread_dataset(file_id, "/in/cpmlzone", H5T_NATIVE_FLOAT, g->cpmlzone); // transfer from hdf5 file
    else
        PrintLog("No CPML zones  defined... ");

    PrintLog("Done ! \n");
    /*for (cnt=0; cnt<g->zczc; cnt++)
        {     printf("\n-----------\n");
        for (int cnt2=0; cnt2<NP_CPML;cnt2++)
        {
                  printf(" %f ",g->cpmlzone[cnt*NP_CPML+cnt2]);}

            }*/

    H5Fclose(file_id); //close the hdf5 file

    return;
}


#endif
