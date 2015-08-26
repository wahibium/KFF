
#ifndef MY_CPML_HELP_H
#define MY_CPML_HELP_H
#include <stdio.h>
#include <unistd.h>
#include "grid.cu"
#include "DebugOutput.h"
#include "cutil_temp.h"// This file was around. I used it because nvcc could not find functions like CUDA_SAFE_CALL, presumebly present in cutil.h
#include "ConstantMemoryInit.h"
#include "bitoperations.cu"
#include "ArrayHandling.cu"
#include "region_distance_help.cu"

#define NEG 1
#define NOPML 0
#define POS 2
#define ELEC 0 // Electric field  
#define MAG 1 // Magnetic field
#define cpmlhost(cnt,f,p) g->cpmlcell[g->ncpmlcells*(f*NP_PF_CPML_CC+NP_FIXED_CPML_CC+p)+cnt]
#define GET_I(d,cnt) d*(THICK_CPML_MAX)+cnt // gets the index of field i in direction d
#define GET_I2(d,cnt) d*THICK_CPML_MAX+THICK_CPML_MAX/2+cnt // gets the index of field i+1/2 in direction d
FILE *debugzone1, *debugzone2, *debugzone3, *debugzone4;

// Function to get the EXACT distance in SI units between cell with index x0 y0 z0 and a cell dx dy dz away.
// The distance is calculated exactly to match the distance between field with componentstart and componentstop(give 6 for the cell center)
// The distance is stored in ansx ansy andz

void get_distance_cpml(my_grid *g, int start, int stop, float xd[2], int direction) {

    int dirE[3] = {H_X, H_Y, H_Z};
    int dirH[3] = {E_X, E_Y, E_Z};
    get_distance_yee(g, start, start, start, stop, stop, stop, NEGATIVE_BORDER, dirE[direction], &xd[0], direction);
    get_distance_yee(g, start, start, start, stop, stop, stop, NEGATIVE_BORDER, dirH[direction], &xd[1], direction);
}

void GetSigmaKappaAlphaMax2(my_grid *g, int zone, float *sigmamax, float* kappamax, float *alphamax) {
    for (int temp = 0; temp < 3; temp++) {
        sigmamax[temp] = g->cpmlzone[zone * NP_CPML + POS_SMAXSTART_CPML];
        kappamax[temp] = g->cpmlzone[zone * NP_CPML + POS_KMAXSTART_CPML];
        alphamax[temp] = g->cpmlzone[zone * NP_CPML + POS_AMAXSTART_CPML];
    }

}

void GetSigmaKappaAlphaMax(my_grid *g, int zone, float *sigmamax, float* kappamax, float *alphamax) {

    sigmamax[0] = g->cpmlzone[zone * NP_CPML + POS_SMAXSTART_CPML ];
    kappamax[0] = g->cpmlzone[zone * NP_CPML + POS_KMAXSTART_CPML ];
    alphamax[0] = g->cpmlzone[zone * NP_CPML + POS_AMAXSTART_CPML ];
}

void GetD(my_grid *g, int zone, float* d) // Gets the actual maximal distance of the CPML
{
    float tempx, tempy, tempz;
    int x = g->cpmlzone[zone * NP_CPML + 1];
    int y = g->cpmlzone[zone * NP_CPML + 2];
    int z = g->cpmlzone[zone * NP_CPML + 3];
    int dx = g->cpmlzone[zone * NP_CPML + 4];
    int dy = g->cpmlzone[zone * NP_CPML + 5];
    int dz = g->cpmlzone[zone * NP_CPML + 6];

    assert(dx >= 0);
    assert(dy >= 0);
    assert(dz >= 0);
    
    // X Direction.
    // Hx is at the plane of incidence when propagating in the x direction
    // We end at the last cell in the X direction.
    get_distance_yee(g, x, y, z, dx, 0, 0, NEGATIVE_BORDER, POSITIVE_BORDER, &tempx, 0);
    d[0] = tempx;

    // Ydirection
    // Hy is at the plane of incidance when propagating in the y direction
    // We end at the last cell in the Y direction.
    get_distance_yee(g, x, y, z, 0, dy, 0, NEGATIVE_BORDER, POSITIVE_BORDER, &tempy, 1);
    d[1] = tempy;

    // Z direction
    // Hz is at the plane of incidance when propagating in the z direction
    // We end at the last cell in the Z direction.
    get_distance_yee(g, x, y, z, 0, 0, dz, NEGATIVE_BORDER, POSITIVE_BORDER, &tempz, 2);
    d[2] = tempz;

}
// detectsif we have a face

void GetSigmaAlphaKappa(float *xd, float *d, float *sigmamax, float *sigmaE,
        float *sigmaH, float *kappamax, float *kappaE, float *kappaH, float *alphamax, float*alphaE, float *alphaH, int m, int direction) {
    // Calculates the the sigma kappas and alphas for an individual cell.
    // santity check
    if (xd[0] < 0) printf("\nCheck calculation of indices CPML, xd[0]=%e,d[direction]=%e", xd[0], d[direction]); // Sanity check
    if (xd[1] < 0) printf("\nCheck calculation of indices CPML xd[1]=%e,d[direction]=%e", xd[1], d[direction]); // Sanity check

    sigmaE[direction] = pow(xd[0] / d[direction], m) * sigmamax[direction]; // Eq. 7.60a
    sigmaH[direction] = pow(xd[1] / d[direction], m) * sigmamax[direction]; // Eq. 7.60a

    kappaE[direction] = 1 + (kappamax[direction] - 1) * pow(xd[0] / d[direction], m); // Eq. 7.60b
    kappaH[direction] = 1 + (kappamax[direction] - 1) * pow(xd[1] / d[direction], m); // Eq. 7.60b

    alphaE[direction] = pow((d[direction] - xd[0]) / d[direction], m) * alphamax[direction]; // Eq. 7.79
    alphaH[direction] = pow((d[direction] - xd[1]) / d[direction], m) * alphamax[direction]; // Eq. 7.79

}

void GetBC(float* sigmaE, float *sigmaH, float*alphaE, float*alphaH, float* kappaE, float*kappaH, float dt, float*bE, float*bH, float*cE, float *cH, int direction) {
    //holds b and c (7.99) and (7.102) for each direction. One used of update of PsiE the other for the update of PsiH
    bE[direction] = exp(-(sigmaE[direction] / (EPSILON_0 * kappaE[direction]) + alphaE[direction] / EPSILON_0) * dt);
    cE[direction] = sigmaE[direction] / (sigmaE[direction] * kappaE[direction] + alphaE[direction] * kappaE[direction] * kappaE[direction])*(exp(-(sigmaE[direction] / (EPSILON_0 * kappaE[direction]) + alphaE[direction] / EPSILON_0) * dt) - 1);
    bH[direction] = exp(-(sigmaH[direction] / (EPSILON_0 * kappaH[direction]) + alphaH[direction] / EPSILON_0) * dt);
    cH[direction] = sigmaH[direction] / (sigmaH[direction] * kappaH[direction] + alphaH[direction] * kappaH[direction] * kappaH[direction])*(exp(-(sigmaH[direction] / (EPSILON_0 * kappaH[direction]) + alphaH[direction] / EPSILON_0) * dt) - 1);
    // correction to avoid divition by zero when sigmaE[direction]==0
    if (sigmaE[direction] == 0) cE[direction] = 0;
    if (sigmaH[direction] == 0) cH[direction] = 0;
}

void PopulateConstantArrays(my_grid *g, int zone)// Populate the constant Arrays.
{
    int dx = g->cpmlzone[zone * NP_CPML + POS_POSDSTART_CPML + 0];
    int dy = g->cpmlzone[zone * NP_CPML + POS_POSDSTART_CPML + 1];
    int dz = g->cpmlzone[zone * NP_CPML + POS_POSDSTART_CPML + 2];
    int m = g->cpmlzone [zone * NP_CPML + POS_M_CPML];
    float sigmamax[3], sigmaE[3], sigmaH[3]; //holds the sigmamaxes in xyz,
    float kappamax[3], kappaE[3], kappaH[3]; // holds the kappamaxes in xyz
    float alphamax[3], alphaE[3], alphaH[3]; // holds the the aplhamax
    float bE[3], bH[3], cE[3], cH[3]; // holds b and c (7.99) and (7.102) for each direction. One used of update of PsiE the other for the update of PsiH
    float dir[3] = {dx, dy, dz};
    float d[3];
    float xd[2]; // temp variables
    GetSigmaKappaAlphaMax2(g, zone, sigmamax, kappamax, alphamax); // get the maxima.
    GetD(g, zone, d);
    for (int direc = 0; direc < 3; direc++) {
        for (int cnt = 0; cnt <= dir[direc]; cnt++) {
            get_distance_cpml(g, 0, cnt, xd, direc);
            GetSigmaAlphaKappa(xd, d, sigmamax, sigmaE, sigmaH, kappamax, kappaE, kappaH, alphamax, alphaE, alphaH, m, direc);
            GetBC(sigmaE, sigmaH, alphaE, alphaH, kappaE, kappaH, g->dt, bE, bH, cE, cH, direc);
            // write down the indices.
            // Store the inverse because less divisions in the kernel.

            g->kappa[GET_I(direc, cnt)] = 1 / kappaE[direc]; //kappaE[direc];
            g->kappa[GET_I2(direc, cnt)] = 1 / kappaH[direc]; //kappaH[direc];
            g->bcpml[GET_I(direc, cnt)] = bE[direc];
            g->bcpml[GET_I2(direc, cnt)] = bH[direc];
            g->ccpml[GET_I(direc, cnt)] = cE[direc];
            g->ccpml[GET_I2(direc, cnt)] = cH[direc];
        }
    }

}

void PopulateBorderCpml(my_grid*g, int zone) {
    int dx = g->cpmlzone[zone * NP_CPML + POS_POSDSTART_CPML + 0];
    int dy = g->cpmlzone[zone * NP_CPML + POS_POSDSTART_CPML + 1];
    int dz = g->cpmlzone[zone * NP_CPML + POS_POSDSTART_CPML + 2];
    int cpmlindex = g->cpmlzone[zone * NP_CPML + POS_CPMLINDEX_CPML];
    int dimsim[3] = {g->xx, g->yy, g->zz};
    int dimcpml[3] = {dx, dy, dz};
    int indexi, indexi2;
    int newcpmlindex = 0; // hashes cpml direction. Writes it into propnum CPML
    int pos[3];
    int orth [3][6] = {
        {0, 1, 1, 0, 1, 1}, //x
        {1, 0, 1, 1, 0, 1}, //y
        {1, 1, 0, 1, 1, 0} //z
    }; 
    for (int i = 0; i < g->xx; i++) {
        for (int j = 0; j < g->yy; j++) {
            for (int k = 0; k < g->zz; k++) { // over all cells.
                newcpmlindex = 0; // By default there is no cpml in any direction
                pos[0] = i;
                pos[1] = j;
                pos[2] = k;
                g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy] = 0;
                g->property[PROPNUM_CPML][(i) + (j) * g->xx + (k) * g->xx * g->yy] = 0;
                for (int dir = 0; dir < 3; dir++) {
                    
                    //x-y-z- (only for last cell (closest to simulation))
                    if ((pos[dir] == dimcpml[dir] - 1) && (cpmlindex & (1 << (2 * (dir) + 1)))) {
                        indexi = GET_I(0, dimcpml[dir] - pos[dir] - 1);

                        // all fields are updated.
                        for (int f = 0; f < 3; f++) {
                            if (orth[dir][f] == 1)
                                g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy] =  \
                                SetMatType(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, CPML);
                        }
                        newcpmlindex = newcpmlindex + (1 << dir); //update newcpmlindex;
                        g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy] =
                                SetCpmlIndex(g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy], dir, indexi);
                    }

                    //x-y-z- (only for 0:dimcpml-1)
                    if ((pos[dir] < dimcpml[dir] - 1) && (cpmlindex & (1 << (2 * (dir) + 1)))) {

                        indexi = GET_I(0, dimcpml[dir] - pos[dir] - 1);
                        indexi2 = GET_I2(0, dimcpml[dir] - pos[dir] - 2);
                        
                        // all fields are updated.
                        for (int f = 0; f < 6; f++) {
                            if (orth[dir][f] == 1)
                                g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy] =  \
                                SetMatType(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, CPML);
                        }

                        newcpmlindex = newcpmlindex + (1 << dir)+(1 << (dir + 3)); //update newcpmlindex;
                        g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy] =
                                SetCpmlIndex(g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy], dir, indexi);
                        g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy] =
                                SetCpmlIndex(g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy], dir + 3, indexi2);

                    }

                    //(x+,y+,z+)
                    if ((pos[dir] >= dimsim[dir] - dimcpml[dir]) && (cpmlindex & (1 << 2 * dir))) {
                        indexi = GET_I(0, pos[dir] - dimsim[dir] + dimcpml[dir]);
                        indexi2 = GET_I2(0, pos[dir] - dimsim[dir] + dimcpml[dir]); // on cell to the right
                        // all fields are updated.

                        for (int f = 0; f < 6; f++) {
                            if (orth[dir][f] == 1)
                                g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy] =  \
                                SetMatType(g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy], f, CPML);
                        }


                        newcpmlindex = newcpmlindex + (1 << dir)+(1 << (dir + 3)); //update newcpmlindex;
                        g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy] =
                                SetCpmlIndex(g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy], dir, indexi);
                        g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy] =
                                SetCpmlIndex(g->property[PROPNUM_CPML_INDEX][(i) + (j) * g->xx + (k) * g->xx * g->yy], dir + 3, indexi2);
                    }
                }

                // Writing the cpmlindex in the hash.
                g->property[PROPNUM_CPML][(i) + (j) * g->xx + (k) * g->xx * g->yy] =   \
                SetPropertyFast(g->property[PROPNUM_CPML][(i) + (j) * g->xx + (k) * g->xx * g->yy], SELECTOR_ISCPML, FIRST_BIT_ISCPML, newcpmlindex);
            }
        }
    }
}

void PopulateDeviceCpml(my_grid*g) // Populates the device mememory for the CPMLs
{
    //Detect how many cells there are to do memory allocation.
    int cnt = 0;
    int cpmltoggle;
    unsigned long matparam;
    for (int k = 0; k < g->zz; k++) {
        for (int j = 0; j < g->yy; j++) {
            for (int i = 0; i < g->xx; i++) {
                // Detect how many cells we have.
                cpmltoggle = 0;
                matparam = g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy];

                for (int f = 0; f < 6; f++) {
                    int type = GetPropertyFastHost(matparam, SELECTOR_MATTYPE << f*BIT_MATTYPE_FIELD, f * BIT_MATTYPE_FIELD);
                    if (type & CPML) {
                        cpmltoggle = 1;
                    }
                }

                if (cpmltoggle) {
                    g->property[PROPNUM_CPML][(i) + (j) * g->xx + (k) * g->xx * g->yy] = \
                 SetPropertyFast(g->property[PROPNUM_CPML][(i) + (j) * g->xx + (k) * g->xx * g->yy], SELECTOR_CPML, FIRST_BIT_CPML, cnt);

                    cnt = cnt + 1;
                }
            }
        }
    }

    // Allocate Memory for the devicecpml.
    g->cpmlcell = (float*) cust_alloc(sizeof (float) * cnt * (6 * NP_PF_CPML_CC));
    g->ncpmlcells = cnt;


    printf("\nCPMLArray contains: %d cells \n", cnt);
    if (g->ncpmlcells >= (int) pow(2, BIT_CPML)) {
        printf("Error:There are too many cpml cells defined in this simulation");
        assert(0);
    }

}

void GetUpdateBorderCpmlZones(my_grid *g) // Goes over all the cpml zones and checks for borders.
//If a border creates a new zone array with the right parameters.
{
    int cnt;
    cnt = 0;
    do {
        if (g->cpmlzone[cnt * NP_CPML + POS_BORDER_CPML] == 1) {//Add the borderzones.
            // Add New zones at the end.
            PopulateConstantArrays(g, cnt); // Populate the constant Arrays.
            PopulateBorderCpml(g, cnt); //populate the border zones.
            PopulateDeviceCpml(g);
            char outfile[512];
            sprintf(outfile, "%s%s", outdir, "debugzone3.txt");
            debugzone3 = fopen(outfile, "w+");
            sprintf(outfile, "%s%s", outdir, "debugzone4.txt");
            debugzone4 = fopen(outfile, "w+");

            // CPML DEBUG

            /* for (int k = 0; k < g->zz; k++) {
                 unsigned long param1 = g->property[PROPNUM_CPML_INDEX][(32) + (32) * g->xx + (k) * g->xx * g->yy];
                unsigned long param2 = g->property[PROPNUM_CPML_INDEX][(32) + (32) * g->xx + (k) * g->xx * g->yy];
                 unsigned long matparam = g->property[PROPNUM_MATTYPE][(32) + (32) * g->xx + (k) * g->xx * g->yy];
                 printbitssimple(param1);
                 int index1;
                 int index2;
                 int mattype1, mattype2;
                 index1 = GetPropertyFastHost(param1, SELECTOR_CPMLINDEX << 2 * BIT_CPMLINDEX_FIELD, 2 * BIT_CPMLINDEX_FIELD);
                 mattype1 = GetPropertyFastHost(matparam, SELECTOR_MATTYPE << 0 * BIT_MATTYPE_FIELD, 0 * BIT_MATTYPE_FIELD);
                 mattype2 = GetPropertyFastHost(matparam, SELECTOR_MATTYPE << 3 * BIT_MATTYPE_FIELD, 3 * BIT_MATTYPE_FIELD);
                 printbitssimple(index1);
                 index2 = GetPropertyFastHost(param2, SELECTOR_CPMLINDEX << 5 * BIT_CPMLINDEX_FIELD, 5 * BIT_CPMLINDEX_FIELD);

                 float out1 = g->bcpml[GET_INDEX(2, index1)];
                 float out2 = g->bcpml[GET_INDEX(2, index2)];

                 //float param1 = xd[0];

                 fprintf(debugzone3, "%e \t%d\t %d\n", out1, k, mattype1);
                 //float param2 = xd[1];
                 fprintf(debugzone4, "%e \t%d\t%d\n", out2, k, mattype2);
             }*/
            //Delete the borderzone
            DeleteZone(g, cnt);
            // restart the count
            cnt = -1;
        }
        cnt = cnt + 1;
    } while (cnt < g->zczc);

}



#endif // MY_CPML_HELP
