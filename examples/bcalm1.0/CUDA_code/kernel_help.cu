#ifndef MY_KERNEL_HELP
#define MY_KERNEL_HELP



#define tx threadIdx.x
#define ty threadIdx.y

#define bx blockIdx.x
#define by blockIdx.y

#define cas(i,j) csource[i*NPSOURCE+j]
#define cacpml(index,f,p) ccpmlcell[ncpmlcells*(f*NP_PF_CPML_CC+p)+index]
#define caf(i,j,k,f) cfield[f][(i) + (j)*cxx+ (k)*cxx*cyy]
#define cap(i,j,k,p) cproperty[p][(i) + (j)*cxx+ (k)*cxx*cyy]
#define cagx(i) cgridx[i]
#define cagy(j) cgridy[j]
#define cagz(k) cgridz[k]
#define maf(i,j,k,f) field[f][(i) + __mul24(j, xx)+ (k)*xx*yy]
#define af(i,j,k,f) g.field[f][(i) + (j)*g.xx+ (k)*g.xx*g.yy] 
#define ae(i,j,k,f) g.epsilon[f][(i) + (j)*g.xx+ (k)*g.xx*g.yy] 
#define am(i,j,k,f) g.mat[f][(i) + (j)*g.xx+ (k)*g.xx*g.yy]
#define ap(i,j,k,p) g.property[p][(i) + (j)*g.xx+ (k)*g.xx*g.yy]
#define as(i,j) g.source[i*NPSOURCE+j]
#define glclC(cnt,tog,pole,param) clorentzfield[nlorentzfields*(NP_PP_LORENTZ_CL * pole*tog + tog*NP_FIXED_LORENTZ_CL+param)+cnt] //In constant Memory
#define glcC(mat,pole,param) clorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + NP_FIXED_LORENTZ_C+ pole*NP_PP_LORENTZ_C +param]//In constant.

#include "populate.cu"



__device__ void my_get_source(float (*p)[MAX_PARAM_NEEDED],unsigned long property,int f)

{unsigned long sourceindex=GetPropertyFast(property,selectorsource,FIRST_BIT_SOURCE);

      (*p)[POS_ABS_SOURCE_D]=cas(sourceindex,POS_ABS_SOURCE+f);//Abs Amplitude
      (*p)[POS_PHASE_SOURCE_D]=cas(sourceindex,POS_PHASE_SOURCE+f);//Phase Amplitude
      (*p)[POS_OMEGA_SOURCE_D]=cas(sourceindex,POS_OMEGA_SOURCE);//Omega (inefficient could be read one level up to save a memory transfer)
      (*p)[POS_MUT_SOURCE_D]=cas(sourceindex,POS_MUT_SOURCE);//ut (inefficient could be read one level up to save a memory transfer)
      (*p)[POS_SIGMAT_SOURCE_D]=cas(sourceindex,POS_SIGMAT_SOURCE);//sigmat(ineffcient could be read one level up to save a memory transfer)
}

__device__ float current_source (float p[MAX_PARAM_NEEDED], int t) {
	return p[POS_ABS_SOURCE_D]*sinf( p[POS_OMEGA_SOURCE_D] * t*cdt+p[POS_PHASE_SOURCE_D]) * expf ( -1 * powf (((t  - p[POS_MUT_SOURCE_D])/p[POS_SIGMAT_SOURCE_D]), 2.0));

}



__device__ void load_fieldE(float Ex[B_XX][B_YY],float Ey[B_XX][B_YY],float Ez[B_XX][B_YY],int i, int j, int k)

{
    Ex[tx][ty] = caf (i,j,k,0);
    Ey[tx][ty] = caf (i,j,k,1);
    Ez[tx][ty] = caf (i,j,k,2);
}





__device__ void load_E (float H[B_XX+1][B_YY+1], int i, int j, int k, int f) {

    H[tx+1][ty+1] = caf (i,j,k,f);
 
	if (tx == 0) {
		if (bx == 0)
			H[tx][ty+1] = 0;
		else
			H[tx][ty+1] = caf (i-1,j,k,f);

		}
	if (ty == 0) {
		if (by == 0)
			H[tx+1][ty] = 0;
		else
			H[tx+1][ty] = caf (i,j-1,k,f);

		}
	return;
}

__device__ void load_H ( float E[B_XX+1][B_YY+1], int i, int j, int k, int f) {
	E[tx][ty] = caf (i,j,k,f);
	
	if (tx == B_XX-1) {
		if (bx == (cxx/B_XX)-1)
			E[tx+1][ty] = 0;
		else
			E[tx+1][ty] = caf (i+1,j,k,f);

		}
	if (ty == B_YY-1) {
		if (by == (cyy/B_YY)-1)
			E[tx][ty+1] = 0;
		else
			E[tx][ty+1] = caf (i,j+1,k,f);
		}
	return;
}

// transfers from A to B
__device__ void transfer_E (float A[B_XX+1][B_YY+1], float B[B_XX+1][B_YY+1]) {
	B[tx+1][ty+1] = A[tx+1][ty+1];
	if (tx == 0)
		B[tx][ty+1] = A[tx][ty+1];
	if (ty == 0)
		B[tx+1][ty] = A[tx+1][ty];
	return;
}

__device__ void transfer_H (float A[B_XX+1][B_YY+1], float B[B_XX+1][B_YY+1]) {
	B[tx][ty] = A[tx][ty];
	if (tx == B_XX-1)
		B[tx+1][ty] = A[tx+1][ty];
	if (ty == B_YY-1)
		B[tx][ty+1] = A[tx][ty+1];
	return;
}
	


__device__ void make_zero(float A[B_XX+1][B_YY+1]) {
	A[tx+1][ty+1] = 0;
	if (tx == 0)
		A[tx][ty+1] = 0;
	if (ty == 0)
		A[tx+1][ty] = 0;
	return;
}

#endif
