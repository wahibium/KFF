// kernel update for CUDA
#ifndef KERNEL_CU 
#define KERNEL_CU 

#include "grid.cu"
#include "kernel_help.cu"
#include "cutil_temp.h" 
#include "populate.cu"
#include "kernel_inline.cu"
#include "DebugOutput.h"

__global__ void kernel_E(int t) {
    int i, j, k;
    unsigned long property[NUM_PROP_TOT];
    //unsigned long property2;
    int mat_type[3];
    // initialize index values
    i = tx + bx*B_XX;
    j = ty + by*B_YY;

    __shared__ float Hx_a[B_XX + 1][B_YY + 1];
    __shared__ float Hy_a[B_XX + 1][B_YY + 1];
    __shared__ float Hz_a[B_XX + 1][B_YY + 1];
    __shared__ float Hx_b[B_XX + 1][B_YY + 1];
    __shared__ float Hy_b[B_XX + 1][B_YY + 1];

    make_zero(Hx_b); // For the first layer in Z we assume Hx and Hy to be zero ---> PMC;
    make_zero(Hy_b);

    for (k = 0; k < czz; k++) {

        load_E(Hx_a, i, j, k, 3); // Load the magnetic fields Hx_a
        load_E(Hy_a, i, j, k, 4); // Load the magnetic fields Hy_a
        load_E(Hz_a, i, j, k, 5); // Load the magnetic fields Hz_a

        __syncthreads(); // wait for all values to be loaded in

        // load the properties
        property[PROPNUM_MATTYPE] = cap(i, j, k, PROPNUM_MATTYPE);
        property[PROPNUM_AMAT] = cap(i, j, k, PROPNUM_AMAT);
        property[PROPNUM_CPML] = cap(i, j, k, PROPNUM_CPML);
        property[PROPNUM_CPML_INDEX] = cap(i, j, k, PROPNUM_CPML_INDEX);

        // load the sources
        if (property[PROPNUM_MATTYPE] & selector_allfield_source[0])
            property[PROPNUM_SOURCE] = cap(i, j, k, PROPNUM_SOURCE);
        // Extract the mat_type
        mat_type[0] = GetPropertyFast(property[PROPNUM_MATTYPE], selectormattype[0], 0 * BIT_MATTYPE_FIELD);
        mat_type[1] = GetPropertyFast(property[PROPNUM_MATTYPE], selectormattype[1], 1 * BIT_MATTYPE_FIELD);
        mat_type[2] = GetPropertyFast(property[PROPNUM_MATTYPE], selectormattype[2], 2 * BIT_MATTYPE_FIELD);
        // update E
        update_E(0, Hx_a, Hx_b, Hy_a, Hy_b, Hz_a, property, mat_type[0], i, j, k, t);
        update_E(1, Hx_a, Hx_b, Hy_a, Hy_b, Hz_a, property, mat_type[1], i, j, k, t);
        update_E(2, Hx_a, Hx_b, Hy_a, Hy_b, Hz_a, property, mat_type[2], i, j, k, t);

        __syncthreads(); // wait for all computation to be finished before loading in new values

        // obtain the bottom layer through a swap operation
        // transfer_E(source, destination)
        transfer_E(Hx_a, Hx_b);
        transfer_E(Hy_a, Hy_b);
    }

    return;
}

__global__ void kernel_H(int t) {
    int i, j, k;
    int mat_type[3];
    unsigned long property[NUM_PROP_TOT];
    // initialize index values
    i = tx + bx*B_XX;
    j = ty + by*B_YY;

    __shared__ float Ex_a[B_XX + 1][B_YY + 1];
    __shared__ float Ey_a[B_XX + 1][B_YY + 1];
    __shared__ float Ez_a[B_XX + 1][B_YY + 1];
    __shared__ float Ex_b[B_XX + 1][B_YY + 1];
    __shared__ float Ey_b[B_XX + 1][B_YY + 1];

    // load the current layer
    load_H(Ex_a, i, j, 0, 0);
    load_H(Ey_a, i, j, 0, 1);

    for (k = 0; k < czz; k++) {

        // load the upper layer
        load_H(Ez_a, i, j, k, 2); // and Ez for the current layer
        if (k == czz - 1) {
            make_zero(Ex_b); // If last layer in Z Ex and Ey of the next layer are assumed to be zero--> PEC boundary
            make_zero(Ey_b); // If last layer in Z Ex and Ey of the next layer are assumed to be zero--> PEC boundary

        } else {
            load_H(Ex_b, i, j, k + 1, 0);
            load_H(Ey_b, i, j, k + 1, 1);

        }
        __syncthreads(); // wait for all values to be loaded in

        // load the current layer
        property[PROPNUM_MATTYPE] = cap(i, j, k, PROPNUM_MATTYPE);
        property[PROPNUM_CPML] = cap(i, j, k, PROPNUM_CPML);
        property[PROPNUM_CPML_INDEX] = cap(i, j, k, PROPNUM_CPML_INDEX);

        if (property[PROPNUM_MATTYPE] & selector_allfield_source[1])
            property[PROPNUM_SOURCE] = cap(i, j, k, PROPNUM_SOURCE);

        mat_type[0] = GetPropertyFast(property[PROPNUM_MATTYPE], selectormattype[3], 3 * BIT_MATTYPE_FIELD);
        mat_type[1] = GetPropertyFast(property[PROPNUM_MATTYPE], selectormattype[4], 4 * BIT_MATTYPE_FIELD);
        mat_type[2] = GetPropertyFast(property[PROPNUM_MATTYPE], selectormattype[5], 5 * BIT_MATTYPE_FIELD);

        update_H(3, Ex_a, Ex_b, Ey_a, Ey_b, Ez_a, property, mat_type[0], i, j, k, t);
        update_H(4, Ex_a, Ex_b, Ey_a, Ey_b, Ez_a, property, mat_type[1], i, j, k, t);
        update_H(5, Ex_a, Ex_b, Ey_a, Ey_b, Ez_a, property, mat_type[2], i, j, k, t);

        __syncthreads(); // wait for all computation to be finished before loading in new values

        // obtain the the new current layer through a swap operation
        transfer_H(Ex_b, Ex_a);
        transfer_H(Ey_b, Ey_a);
    }

    return;
}

//Kernel to be executed AFTER kernelE

__global__ void kernel_lorentz() {

    float alphaP[N_POLES_MAX], xiP[N_POLES_MAX], gammaP[N_POLES_MAX];
    float J[N_POLES_MAX], JN1[N_POLES_MAX];
    float Enm1, En, Entemp, Enp1, C1, C3, lastterm, Jnp1;
    int cnt = tx + bx*B_XX_LORENTZ, np, mat; // Get the threadID of the lorentzcells.
    float* fieldaddress = clorentzfieldaddress[cnt];

    if (fieldaddress == NULL) return; // Check if we are at the end of our array.

    // Load the material number.
    mat = glclC(cnt, 0, 0, POS_INDEX_LORENTZ_CL);
    np = clorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_NP_LORENTZ_C];
    // Load C1,C3
    C1 = clorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C1_LORENTZ_C];
    C3 = clorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C3_LORENTZ_C];

    //Load the material properties and the pole currents.
    for (int pole = 0; pole < np; pole++) {
        alphaP[pole] = glcC(mat, pole, POS_ALPHAP_LORENTZ_C);
        xiP [pole] = glcC(mat, pole, POS_XIP_LORENTZ_C);
        gammaP[pole] = glcC(mat, pole, POS_GAMMAP_LORENTZ_C);
    }
    // Loading the fields
    En = glclC(cnt, 0, 0, POS_FIELD_LORENTZ_CL);
    Enm1 = glclC(cnt, 0, 0, POS_FIELD_LORENTZ_CL + 1);
    //Load of the electrical field from the grid (non coalesced).Define structures in the X direction
    Entemp = *fieldaddress; //=C2En+C3(Rot(H)+psiCPML);

    lastterm = 0;
    //caculate the last term of (9.52)
    for (int pole = 0; pole < np; pole++) {
        J[pole] = glclC(cnt, 1, pole, POS_J_LORENTZ_CL);
        JN1[pole] = glclC(cnt, 1, pole, POS_J_LORENTZ_CL + 1);
        lastterm = lastterm + 0.5 * ((1 + alphaP[pole]) * J[pole] + xiP[pole] * JN1[pole]);
    }
    // Calculate the final Ep1
    Enp1 = Entemp + C1 * Enm1 - C3*lastterm;


    //Write it in the field array.
    *fieldaddress = Enp1;


    //Write En=En+1

    glclC(cnt, 0, 0, POS_FIELD_LORENTZ_CL) = Enp1;
    //Write En-1=En
    glclC(cnt, 0, 0, POS_FIELD_LORENTZ_CL + 1) = En;


    //Calculate deltaE
    float dE = (Enp1 - Enm1) / (2 * cdt);

    //Updating the JN
    for (int pole = 0; pole < np; pole++) {
        Jnp1 = alphaP[pole] * J[pole] + xiP[pole] * JN1[pole] + gammaP[pole] * dE;

        glclC(cnt, 1, pole, POS_J_LORENTZ_CL + 1) = J[pole]; // Jpn-1=Jpn
        glclC(cnt, 1, pole, POS_J_LORENTZ_CL) = Jnp1;


    }

}

#endif // MY_KERNEL_CU
