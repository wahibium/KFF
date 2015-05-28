 #ifndef MY_INLINE_CU
#include "cpml_help.cu"
#include "DebugOutput.h"

__device__ inline void update_E(int component,
        float Hx_a[B_XX + 1][B_YY + 1],
        float Hx_b[B_XX + 1][B_YY + 1],
        float Hy_a[B_XX + 1][B_YY + 1],
        float Hy_b[B_XX + 1][B_YY + 1],
        float Hz_a[B_XX + 1][B_YY + 1],
        unsigned long* property, int mat_type, int i0, int j0, int k0, int t) {
    int i, j, index, cpmlindex;
    unsigned long pindex, pcpmlindex;
    float kappat[3];
    int findex1, findex2;
    i = threadIdx.x + 1;
    j = threadIdx.y + 1;

    float E, epsilon, EC1, EC2, psiE1 = 0, psiE2 = 0;
    float H1A, H1B, H2A, H2B, cgrid1, cgrid2;
    int od1, od2;
    E = caf(i0, j0, k0, component);

    float p[MAX_PARAM_NEEDED];

    // Add the excitation if we are a source
    if (mat_type & SOURCE) {
        my_get_source(&p, property[PROPNUM_SOURCE], component);
        index = GetPropertyFast(property[PROPNUM_DEPS], selectordeps, FIRST_BIT_DEPS);
        //epsilon = deps[index];
        E += current_source(p, t);
    }


    // Get the index of the material(Lorentz or dielectric)
    index = GetPropertyFast(property[PROPNUM_AMAT], selectoramat[component], component * BIT_AMAT_FIELD);

    if (mat_type & LORENTZ) {
        // Set EC1 & EC2 in case of Lorentz materials
        EC1 = clorentzreduced[index * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C2_LORENTZ_C];
        EC2 = clorentzreduced[index * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + POS_C3_LORENTZ_C];
    } else 
    {
        EC1 = C1[index];
        EC2 = C2[index];
    }




    kappat[0] = 1;
    kappat[1] = 1;
    kappat[2] = 1;

    switch (component) {
        case 0:
            H1A = Hz_a[i][j];
            H1B = Hz_a[i][j - 1];
            H2A = Hy_a[i][j];
            H2B = Hy_b[i][j];
            cgrid1 = cgridEy[j0];
            cgrid2 = cgridEz[k0];
            od1 = 1;
            od2 = 2;
            break;
        case 1:
            H1A = Hx_a[i][j];
            H1B = Hx_b[i][j];
            H2A = Hz_a[i][j];
            H2B = Hz_a[i - 1][j];
            cgrid1 = cgridEz[k0];
            cgrid2 = cgridEx[i0];
            od1 = 2;
            od2 = 0;
            break;
        case 2:
            H1A = Hy_a[i][j];
            H1B = Hy_a[i - 1][j];
            H2A = Hx_a[i][j];
            H2B = Hx_a[i][j - 1];
            cgrid1 = cgridEx[i0];
            cgrid2 = cgridEy[j0];
            od1 = 0;
            od2 = 1;
            break;
    }


    if (mat_type & CPML) {
        pindex = property[PROPNUM_CPML];
        pcpmlindex = property[PROPNUM_CPML_INDEX];
        findex1 = GetPropertyFast(pcpmlindex, selectorcpmlindex[od1], od1 * BIT_CPMLINDEX_FIELD);
        findex2 = GetPropertyFast(pcpmlindex, selectorcpmlindex[od2], od2 * BIT_CPMLINDEX_FIELD);

        index = GetPropertyFast(pindex, selectorcpml, FIRST_BIT_CPML);
        cpmlindex = GetPropertyFast(pindex, selectoriscpml, FIRST_BIT_ISCPML);



        if ((cpmlindex >> od1) & 1) {
            psiE1 = bcpml[GET_INDEX(od1, findex1)] * cacpml(index, component, POS_PSIN1_CPML_CC) +
                    ccpml[GET_INDEX(od1, findex1)] *(H1A - H1B) * (cgrid1);
            cacpml(index, component, POS_PSIN1_CPML_CC) = psiE1;
            kappat[od1] = kappa[GET_INDEX(od1, findex1)];
        }
        if ((cpmlindex >> od2) & 1) {
            psiE2 = bcpml[GET_INDEX(od2, findex2)] * cacpml(index, component, POS_PSIN2_CPML_CC) +
                    ccpml[GET_INDEX(od2, findex2)] *(H2A - H2B) * (cgrid2);
            cacpml(index, component, POS_PSIN2_CPML_CC) = psiE2;
            kappat[od2] = kappa[GET_INDEX(od2, findex2)];
        }
    }

    // Update Ex
    // Ex = EC1 * Ex +
    //      EC2 * ((Hz-cellBackY->Hz)/(KyE*(*dy)) 
    //           - (Hy-cellBackZ->Hy)/(KzE*(*dz))
    //           + psiEx1 - psiEx2);
    E = EC1 * E + EC2 * ((H1A - H1B) * (kappat[od1] * cgrid1) -
            (H2A - H2B) * (kappat[od2] * cgrid2) +
            psiE1 - psiE2);

    // Force field to zero if it is a perfectlayer
    if (mat_type & PERFECTLAYER)
        E = 0;


    // write out to global memory
    caf(i0, j0, k0, component) = E;


    return;
}

__device__ inline void update_H(int component,
        float Ex_a[B_XX + 1][B_YY + 1],
        float Ex_b[B_XX + 1][B_YY + 1],
        float Ey_a[B_XX + 1][B_YY + 1],
        float Ey_b[B_XX + 1][B_YY + 1],
        float Ez_a[B_XX + 1][B_YY + 1],
        unsigned long* property, int mat_type, int i0, int j0, int k0, int t) {
    int i, j, index, cpmlindex;
    unsigned long pcpmlindex;
    float kappat[3];
    int findex1, findex2;
    i = threadIdx.x;
    j = threadIdx.y;

    float H, HC = cdt*MU_0INV, psiH1 = 0, psiH2 = 0;
    float E1A, E1B, E2A, E2B, cgrid1, cgrid2;
    int od1, od2;

    H = caf(i0, j0, k0, component);

    float p[MAX_PARAM_NEEDED];


    // Add the excitation if we are a source
    if (mat_type & SOURCE) {
        my_get_source(&p, property[PROPNUM_SOURCE], component);
        H += current_source(p, t);
    }


    kappat[0] = 1;
    kappat[1] = 1;
    kappat[2] = 1;

    switch (component) {
        case 3:
            E1A = Ez_a[i][j + 1];
            E1B = Ez_a[i][j];
            E2A = Ey_b[i][j];
            E2B = Ey_a[i][j];
            cgrid1 = cgridHy[j0];
            cgrid2 = cgridHz[k0];
            od1 = 1;
            od2 = 2;
            break;
        case 4:
            E1A = Ex_b[i][j];
            E1B = Ex_a[i][j];
            E2A = Ez_a[i + 1][j];
            E2B = Ez_a[i][j];
            cgrid1 = cgridHz[k0];
            cgrid2 = cgridHx[i0];
            od1 = 2;
            od2 = 0;
            break;
        case 5:
            E1A = Ey_a[i + 1][j];
            E1B = Ey_a[i][j];
            E2A = Ex_a[i][j + 1];
            E2B = Ex_a[i][j];
            cgrid1 = cgridHx[i0];
            cgrid2 = cgridHy[j0];
            od1 = 0;
            od2 = 1;
            break;
    }


    if (mat_type & CPML) {
        cpmlindex = GetPropertyFast(property[PROPNUM_CPML], selectoriscpml, FIRST_BIT_ISCPML);
        index = GetPropertyFast(property[PROPNUM_CPML], selectorcpml, FIRST_BIT_CPML);
        pcpmlindex = property[PROPNUM_CPML_INDEX];
        findex1 = GetPropertyFast(pcpmlindex, selectorcpmlindex[od1 + 3], (od1 + 3) * BIT_CPMLINDEX_FIELD);
        findex2 = GetPropertyFast(pcpmlindex, selectorcpmlindex[od2 + 3], (od2 + 3) * BIT_CPMLINDEX_FIELD);

        // Update PSI
        if ((cpmlindex >> (od1 + 3) & 1)) {
            psiH1 = bcpml[GET_INDEX(od1, findex1)] * cacpml(index, component, POS_PSIN1_CPML_CC) +
                    ccpml[GET_INDEX(od1, findex1)] *(E1A - E1B) * (cgrid1);
            cacpml(index, component, POS_PSIN1_CPML_CC) = psiH1;
            kappat[od1] = kappa[GET_INDEX(od1, findex1)];
            //psiH1=0;
        }
        if ((cpmlindex >> (od2 + 3) & 1)) {
            psiH2 = bcpml[GET_INDEX(od2, findex2)] * cacpml(index, component, POS_PSIN2_CPML_CC) +
                    ccpml[GET_INDEX(od2, findex2)] *(E2A - E2B) * (cgrid2);
            cacpml(index, component, POS_PSIN2_CPML_CC) = psiH2;
            kappat[od2] = kappa[GET_INDEX(od2, findex2)];



        }
    }

    //Update Hx
    // Hx = Hx - 
    //      HC * ((cellFrontY->Ez-Ez)/(KyH*(*hdy))
    //          - (cellFrontZ->Ey-Ey)/(KzH*(*hdz))
    //          + psiHx1 - psiHx2);
    H = H - HC * ((E1A - E1B) * (kappat[od1] * cgrid1) -
            (E2A - E2B) * (kappat[od2] * cgrid2)
            + psiH1 - psiH2);

    // Force field to zero if it is a perfectlayer
    if (mat_type & PERFECTLAYER)
        H = 0;

    caf(i0, j0, k0, component) = H;

    return;
}





#endif
