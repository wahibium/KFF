#ifndef TRANSFER_CU
#define TRANSFER_CU

#include <stdio.h>
#include "grid.cu"
//#include <cutil.h>
#include "cutil_temp.h"// This file was around. I used it because nvcc could not find functions like CUDA_SAFE_CALL, presumebly present in cutil.h
#include "ConstantMemoryInit.h"
#include "populate.cu"

//__device__ float *d_field[6];

// copy an array into device memory

void* my_copy2device(void *h_data, unsigned int memsize) {
    // allocate device memory
    cudaError_t Errorout;
    void *d_data;
    Errorout = CUDA_SAFE_CALL(cudaMalloc((void**) & d_data, memsize));
    if (Errorout != 0) {
        printf("\n There is an error in the allocating of data of size %d", memsize);
        printf("\n There is an error: %s\n", cudaGetErrorString(Errorout));
        assert(0);
    }

    // copy node structure to device
    Errorout = CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, memsize, cudaMemcpyHostToDevice));
    if (Errorout != 0) {
        printf("\n There is an error in the copying of data of size %d", memsize);
        printf("\n There is an error: %s\n", cudaGetErrorString(Errorout));
        assert(0);
    }
    return d_data;
}

// Copy to contstant memory

void* my_copy2constant(void* d_data, void *h_data, unsigned int memsize, int start) {

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, h_data, memsize, start, cudaMemcpyHostToDevice));
    return d_data;
}

// This function will populate the property array according to all the boxes that it has received

float* copylorentzfielddevice(my_grid g) { // Counting how many cells are out there.

    int np_tot_lorentz_cl = (NP_FIXED_LORENTZ_CL + g.max_lorentz_poles * NP_PP_LORENTZ_CL); // Redefine because macro wants a pointer to g
    return (float*) my_copy2device(g.lorentzfield, sizeof (float) * g.nlorentzfields * np_tot_lorentz_cl);


}

float* copycpmlcelldevice(my_grid g) { // Counting how many cells are out there.

    // g.lorentzcell = (float**)  my_copy2device (g.lorentzcell,sizeof (float*) * ncells);

    return (float*) my_copy2device(g.cpmlcell, sizeof (float) * g.ncpmlcells * (6 * NP_PF_CPML_CC));


}




// Creates a copy of the grid structure which can be used by the kernel functions which run on the device.
// Basically a structure whose pointers point to arrays in device global memory, instead of host memory.
// Split in two parts because we need the address of the fields for the lorentzcell array

my_grid grid2device_field(my_grid g) {
    int cnt;

    //unsigned int memleft;
    //CUdevice dev;
    for (cnt = 0; cnt < 6; cnt++) {
        g.field[cnt] = (float*) my_copy2device(g.field[cnt], sizeof (float) * (g.xx * g.yy * g.zz));

    }


    return g;
}

my_grid grid2device_rest(my_grid h_g, my_grid d_g) {
    int cnt;

    for (cnt = 0; cnt < NUM_PROP_TOT; cnt++) {
        d_g.property[cnt] = (unsigned long*) my_copy2device(h_g.property[cnt], sizeof (unsigned long) *(h_g.xx * h_g.yy * h_g.zz));
    }
    d_g.source = (float*) my_copy2device(h_g.source, sizeof (float) * h_g.ss * NPSOURCE);



    if (h_g.ncpmlcells > 0)
        d_g.cpmlcell = copycpmlcelldevice(h_g);
    if (h_g.nlorentzfields > 0) {
        d_g.lorentzfield = copylorentzfielddevice(h_g);
        d_g.lorentzfieldaddress = (float**) my_copy2device(h_g.lorentzfieldaddress, sizeof (float*) * h_g.nlorentzfields);
    }
    return d_g;
}

void Copy2ConstantMem(my_grid g) // Copies all the necesarry variables to constant memory.
{
    cudaError_t Errorout;
    //float test[NDEPS];
    //printf("%f",g.deps[0]);
    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("C1", g.C1, sizeof (float) * NDEPS, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of C1 Error1: %s\n", cudaGetErrorString(Errorout));

     Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("C2", g.C2, sizeof (float) * NDEPS, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of C1 Error1: %s\n", cudaGetErrorString(Errorout));


    //     Errorout=CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&test, "deps", sizeof(float)*NDEPS,0,cudaMemcpyDeviceToHost ));
    //   printf("Error2: %d",Errorout);
    // for(int i=0;i<NDEPS;i++)
    //printf("\nhejdio------%f--------\n",test[i]);

}

void Copy2ConstantMemD(my_grid gH, my_grid gD) {

    cudaError_t Errorout;

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cxx", &gH.xx, sizeof (int), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cxx Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cyy", &gH.yy, sizeof (int), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cyy Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("czz", &gH.zz, sizeof (int), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of czz Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cdt", &gH.dt, sizeof (float), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cdt Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("csource", &gD.source, sizeof (float*), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of csource Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cfield", &gD.field, sizeof (float*[6]), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of csource Error1: %s\n", cudaGetErrorString(Errorout));


    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cproperty", &gD.property, sizeof (unsigned long*[NUM_PROP_TOT]), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cproperty Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridx", gH.gridX, sizeof (float) * gH.xx, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridx Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridy", gH.gridY, sizeof (float) * gH.yy, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridy Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridz", gH.gridZ, sizeof (float) * gH.zz, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridz Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridEx", gH.gridEx, sizeof (float) * gH.xx, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridx Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridEy", gH.gridEy, sizeof (float) * gH.yy, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridy Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridEz", gH.gridEz, sizeof (float) * gH.zz, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridy Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridHx", gH.gridHx, sizeof (float) * gH.xx, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridx Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridHy", gH.gridHy, sizeof (float) * gH.yy, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridy Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("cgridHz", gH.gridHz, sizeof (float) * gH.zz, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of cgridy Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("clorentzreduced", gH.lorentzreduced, sizeof (float) *(NP_FIXED_LORENTZ_C + NP_PP_LORENTZ_C * N_POLES_MAX) * NLORENTZ, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of clorentzreduced Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("clorentzfield", &gD.lorentzfield, sizeof (float*), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of clorentzcell Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("clorentzfieldaddress", &gD.lorentzfieldaddress, sizeof (int*), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of clorentzcell Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("nlorentzfields", &gH.nlorentzfields, sizeof (float), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of ccpmlcell Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("ccpmlcell", &gD.cpmlcell, sizeof (float*), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of ccpmlcell Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("ncpmlcells", &gH.ncpmlcells, sizeof (float), 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of ccpmlcell Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("kappa", &gH.kappa, sizeof (float) *3 * THICK_CPML_MAX, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of kappa Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("bcpml", &gH.bcpml, sizeof (float) *3 * THICK_CPML_MAX, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of bcpml Error1: %s\n", cudaGetErrorString(Errorout));

    Errorout = CUDA_SAFE_CALL(cudaMemcpyToSymbol("ccpml", &gH.ccpml, sizeof (float) *3 * THICK_CPML_MAX, 0, cudaMemcpyHostToDevice));
    if (Errorout != 0)
        printf("\n There is an error in the copying of ccpml Error1: %s\n", cudaGetErrorString(Errorout));







    // Errorout=CUDA_SAFE_CALL(cudaMemcpyFromSymbol(csourcetest, "selectormattype", sizeof(unsigned long[6]),0,cudaMemcpyDeviceToHost ));
    //printf("Error2: %d",Errorout);
    //printbitssimple(csourcetest[0]);
    //printbitssimple(csourcetest[1]);
    //printf("lala %d",csourcetest[0]);
    //printf("lala %d",csourcetest[1]);
}

void FreeLocalGrid(my_grid* g)
// Frees parts of the unused allocated memory on the CPU ram
// as the grid is copied on the GPU ram.
{
    free(g->gridEx); // Free the field  grids on CPU.
    free(g->gridEy); // Free the fields grids on CPU.
    free(g->gridEz); // Free the fields grids on CPU.
    free(g->gridHx); // Free the fields grids on CPU.
    free(g->gridHy); // Free the fields grids on CPU.
    free(g->gridHz); // Free the fields grids on CPU.
    for (int cnt = 0; cnt < 6; cnt++) {
        free(g->field[cnt]); // Free the fields on CPU.
    }
    free(g->source); // Free the sources on CPU.
    free(g->dielzone); // Free the dielzones.
    free(g->lorentz); // Lorentzstuff on CPU.
    free(g->lorentzfield); // All the lorentzcells on CPU.
    free(g->lorentzfieldaddress); // Address of the lorentzcell on CPU.
    free(g->lorentzreduced); // Reduced Lorentzarray on CPU.
    free(g->lorentzzone); // Free the cpmlzones;



    free(g->cpmlcell); // All the CPML cells on CPU.
    free(g->cpmlzone); // Free the cpmlzones;

    for (int cnt = 0; cnt < NUM_PROP_TOT; cnt++) {
        free(g->property[cnt]); // Free the property array on CPU.
    }
}

void FreeCudaGrid(my_grid* g,my_grid h_g) {
    cudaError_t Errorout;
    int cnt;
    //Freeing the fields.
    for (cnt = 0; cnt < 6; cnt++) {
        Errorout = cudaFree(g->field[cnt]);
        if (Errorout != 0) {
            printf("\n There is an error Freeing the fields on the card: %s\n", cudaGetErrorString(Errorout));
        }
    }

    // Freeing the property array
    for (cnt = 0; cnt < NUM_PROP_TOT; cnt++) {
        Errorout = cudaFree(g->property[cnt]);
        if (Errorout != 0) {
            printf("\n There is an error Freeing the Propertyarray on the card: %s\n", cudaGetErrorString(Errorout));
        }
    }
    // Freeing the Sources
    Errorout = cudaFree(g->source);
    if (Errorout != 0) {
        printf("\n There is an error Freeing the Sources on the card: %s\n", cudaGetErrorString(Errorout));
    }



    if (h_g.ncpmlcells > 0) {
        Errorout = cudaFree(g->cpmlcell);
        if (Errorout != 0) {
            printf("\n There is an error Freeing the Cpmlcells on the card: %s\n", cudaGetErrorString(Errorout));
        }
    }

    if (h_g.nlorentzfields > 0) {

        Errorout = cudaFree(g->lorentzfield);
        if (Errorout != 0) {
            printf("\n There is an error Freeing the Lorentzfields on the card: %s\n", cudaGetErrorString(Errorout));
        }

        Errorout = cudaFree(g->lorentzfieldaddress);
        if (Errorout != 0) {
            printf("\n There is an error Freeing the lorentzfieldaddress on the card: %s\n", cudaGetErrorString(Errorout));
        }

    }




}



#endif // TRANSFER_CU
