


#ifndef CONSTANTINIT
#define CONSTANTINIT

#define B_XX 16
#define B_YY 16
#define B_XX_LORENTZ 256
//--------Constants------------
#define EPSILON_0 8.854187817E-12
#define MU_0  1.256637061435917e-06
#define MU_0INV 7.957747154594768e+05
//--------Max. Size of the Simulation------------
#define SIMUL_SIZE_X 2048
#define SIMUL_SIZE_Y 1024
#define SIMUL_SIZE_Z 1024
//---- Region types--------
#define BOX        0  // fills the whole volume fills a box between x y z and x+dx y+dy z+dz
#define CYLINDER_X 1 // circle in yz cilinder in x fills a box between x y z and x+dx y+dy z+dz
#define CYLINDER_Y 2// circle in xz cilinder in y fills a box between x y z and x+dx y+dy z+dz
#define CYLINDER_Z 3// circle in xy cilinder in z fills a box between x y z and x+dx y+dy z+dz
#define BALL       4 // spehrical in the 3 directions fills a box between x y z and x+dx y+dy z+dz
//------Outs-----------------
#define NP_OUTPUTZONE 17
#define POS_START_O 0 //x,y,z
#define POS_DSTART_O 3 //dx,dy,dz
#define POS_DELTAT_O 6
#define POS_FOUTSTART_O 7
#define POS_FOUTSTOP_O 8
#define POS_FIELD_O  9//6 integers zero or one to know weather to start.
#define POS_DELTAFMIN_0 15
#define POS_AVERAGE 16 // O or 1 to bring back the fields at zone center.
#define MAX_CHAR_OUT 1024 // Num of character per out.1 for the null pointer
                
// Property Array
// PROPNUM_DEPS : 32 bits (for source index, dielectric index and lorentz index)
//   BIT_DEPS: 8 bits (deprecated, populated, but not used in kernel)
//   SOURCES: 18 bits
//   BIT_LORENTZ: 6 bits (deprecated, populated, but not used in kernel)
//     (BIT_DEPS and BIT_LORENTZ were replaced by BIT_AMAT from PROPNUM_AMAT)

// PROPNUM_MATTYPE: 32 bits (for cell material type)
//   MATTYPE: 4 x 6 bits = 24 bits

// PROPNUM_CPML: 32 bits (for cpml addresses of temporary variables)
//

// PROPNUM_CPML_INDEX: 32 bits (for position in the cpml array)
//   BIT_CPMLINDEX_FIELD: 5 x 6 bits = 30 bits

// PROPNUM_AMAT: 32 bits (for anisotropic materials)
//   BIT_AMAT: 3 x 10 bits = 30 bits (index of the material for each field)
//     (decision on dielectric or lorentz is read from MATTYPE)

#define NUM_PROP_TOT 5 // Total property numbers.
#define HALF_WARP_SIZE 16 // Size of a half warp

//--------Mattype-----------------
#define BIT_MATTYPE_FIELD 4 // Number of bits allocated per field.
#define BIT_MATTYPE (BIT_MATTYPE_FIELD*6) //defines the bits to address (4 bits per field)
#define FIRST_BIT_MATTYPE 0 // defines the first bit of the mattype in the property field
#define SELECTOR_MATTYPE ((((unsigned  long) 1)<<(BIT_MATTYPE_FIELD))-1)<<FIRST_BIT_MATTYPE
#define PROPNUM_MATTYPE 0 // Stored in the first property array
//------Matttype(Different Types per field)
#define SOURCE 1 // Index for a source
#define PERFECTLAYER 2 // Index of a perfect layer.
#define LORENTZ 4 // Index of a lorentzcell
#define CPML 8// CPML
#define SELECTOR_ALLFIELD_CPML_E   ((CPML<<0*BIT_MATTYPE_FIELD)|(CPML<<1*BIT_MATTYPE_FIELD)|(CPML<<2*BIT_MATTYPE_FIELD))
#define SELECTOR_ALLFIELD_CPML_H   ((CPML<<3*BIT_MATTYPE_FIELD)|(CPML<<4*BIT_MATTYPE_FIELD)|(CPML<<5*BIT_MATTYPE_FIELD))
#define SELECTOR_ALLFIELD_SOURCE_E ((SOURCE<<0*BIT_MATTYPE_FIELD)|(SOURCE<<1*BIT_MATTYPE_FIELD)|(SOURCE<<2*BIT_MATTYPE_FIELD))
#define SELECTOR_ALLFIELD_SOURCE_H ((SOURCE<<3*BIT_MATTYPE_FIELD)|(SOURCE<<4*BIT_MATTYPE_FIELD)|(SOURCE<<5*BIT_MATTYPE_FIELD))
//------------Dielectric-----------
#define BIT_DEPS 8 // Number of bits allocated for the material type.
#define FIRST_BIT_DEPS (0)// First bit for the dielectric properties in the property field
#define NDEPS 1<<BIT_DEPS-1 // Number of dielectric materials at our disposal
#define NPDIELZONE 8 // Number of parameters to set the properties of of a zone
#define POS_TYPE_DEPS 7 // Where we get the type;
#define SELECTOR_DEPS  ((((unsigned  long) 1)<<(BIT_DEPS))-1)<<FIRST_BIT_DEPS
#define PROPNUM_DEPS 4
//---------AnisotropicMaterial--------
#define BIT_AMAT_FIELD 10 // Number of bits for each electric field.
#define BIT_AMAT (BIT_AMAT_FIELD*3) //Total number of bits per cell to encode for the annisotropic materials
#define FIRST_BIT_AMAT 0 // defines the first bit of the annisotropic material in the property field
#define SELECTOR_AMAT ((((unsigned  long) 1)<<(BIT_AMAT_FIELD))-1)<<FIRST_BIT_AMAT
#define PROPNUM_AMAT 1 // Stored in the first property array
//------------Sources---------------
#define BIT_SOURCE 18 // Number of bits allowed for a source
#define FIRST_BIT_SOURCE (FIRST_BIT_DEPS+BIT_DEPS)// First bit for the dielectric properties in the property field
#define NPSOURCE 18  //Number of floats at our disposal to charctrize our source
#define POS_ABS_SOURCE 3 // Index of the first real amplitude Ex
#define POS_PHASE_SOURCE 9// Index of the first complex amplitude Ex
#define POS_OMEGA_SOURCE 15   // Index of the angular frequency of the source
#define POS_MUT_SOURCE 16  //Index of timestep of the gaussian envelope.
#define POS_SIGMAT_SOURCE 17 //Index of the temporal sigma of the gaussian envelope.
#define SELECTOR_SOURCE  ((((unsigned  long) 1)<<(BIT_SOURCE))-1)<<FIRST_BIT_SOURCE // selector that is going to select the bit in the number
#define PROPNUM_SOURCE 4
//----------Lorentz----------------
#define N_POLES_MAX 4 // Maximum amount of poles
#define BIT_LORENTZ 6// Number of bits allocated for the lorentz type
#define NLORENTZ (1<<BIT_LORENTZ) // Number of lorentz materials at our disposal
#define FIRST_BIT_LORENTZ (FIRST_BIT_SOURCE+BIT_SOURCE)// First bit for the lorentz properties in the property field
#define PROPNUM_LORENTZ 4
#define SELECTOR_LORENTZ ((((unsigned  long) 1)<<(BIT_LORENTZ))-1)<<FIRST_BIT_LORENTZ // selector that is going to select the bit in the number
#define NPLORENTZZONE 7 // Number of parameters to set the properties of of a zone
//------LorentzonHost with original pole parameters
#define NP_FIXED_LORENTZ 3 //fixed parameters to describe the lorentzcell :epsilon and sigma and number of poles
#define NP_PP_LORENTZ 3// Number of paramerters used to describe the lorentzcell per pole.
#define POS_EPS_LORENTZ 0 // Epsilon
#define POS_SIGMA_LORENTZ 1 // The Conductivity sigma
#define POS_NP_LORENTZ 2 // Number of poles used to decribe this pole
#define POS_PD_LORENTZ 3 // Index where pole dependent properties start
#define POS_OMEGAPM_LORENTZ 0 // position of pole depentdent properties after POS_PD_LORENTZ
#define POS_OMEGAM_LORENTZ 1 // Plasma frequency
#define POS_GAMMAM_LORENTZ 2 // Damping
#define gld(mat,pole,param) g->lorentz[mat * (NP_PP_LORENTZ * g->mp + NP_FIXED_LORENTZ) + NP_FIXED_LORENTZ+ pole*NP_PP_LORENTZ +param]//address the correct index.
//------Lorentz in constant memory
#define POS_C1_LORENTZ_C 1 // C1 in constantmem
#define POS_C2_LORENTZ_C 3// C2 in constantmem
#define POS_C3_LORENTZ_C 2// C3 in constantmem
#define POS_NP_LORENTZ_C 0 // Number of poles
#define NP_FIXED_LORENTZ_C 4 //fixed parameters to describe the lorentzcell :C1 C2 C3
#define POS_PP_LORENTZ_C 4// Start of the pole specific parameters in constant memory
#define NP_PP_LORENTZ_C 3 // Number of parameters need per pole in constant memory.
#define POS_ALPHAP_LORENTZ_C 0 // Position of the Alphap after the pole start needed in constantmem
#define POS_XIP_LORENTZ_C 1 // Position of the xip after the pole start in constantmem
#define POS_GAMMAP_LORENTZ_C 2 //Position of gammap after the pole start.
#define glc(mat,pole,param) g->lorentzreduced[mat * (NP_PP_LORENTZ_C * N_POLES_MAX + NP_FIXED_LORENTZ_C) + NP_FIXED_LORENTZ_C+ pole*NP_PP_LORENTZ_C +param]//address the correct index.
//----------Lorentz field array
#define NP_FIXED_LORENTZ_CL 3 //  1 index 2 fields
#define NP_PP_LORENTZ_CL 2 // Jxn-1p Jxp....Jzn-1p Jzp
#define NP_TOT_LORENTZ_CL (NP_FIXED_LORENTZ_CL+g->max_lorentz_poles*NP_PP_LORENTZ_CL)
#define POS_INDEX_LORENTZ_CL 0 // where the material index comes
#define POS_FIELD_LORENTZ_CL 1//2 fields in total Enx En-1x
#define POS_J_LORENTZ_CL 0// Pos Jxp Jxn-1p follows
//(1 cnt of each field)
#define glcl(cnt,tog,pole,param) g->lorentzfield[g->nlorentzfields*(NP_PP_LORENTZ_CL * pole*tog + (tog)*NP_FIXED_LORENTZ_CL+param)+cnt] //address the correct index.
// -----------CPML-----------------------
//-----CPMLHost---- Parameters we get from matlab.
#define NP_CPML 13
#define POS_POSSTART_CPML 1 // xstart ystart zstart
#define POS_POSDSTART_CPML 4 // dx dy dz
#define POS_CPMLINDEX_CPML 0 // Encodes the direction of the PML and its direction [{'x+'},{'x-'},{'y+'},{'y-'},{'z+'},{'z-'}];
#define POS_M_CPML 7 // Exponent to calculate the signas alphas and k
#define POS_SMAXSTART_CPML 8 // smaxx smaxy smaxz
#define POS_KMAXSTART_CPML 9 // kmaxxx kmaxy kmaxz
#define POS_AMAXSTART_CPML 10 // amaxx amaxy amaxz
#define POS_BORDER_CPML 11 // tells weather to be treated as a border or not
#define POS_TERMINATION_CPML 12 // tells in border cases, how the PML should be terminated.
#define PEC_BCPML 0 // PEC termination
#define PMC_BCPML 1 // PMC termination
#define NOTERM_BCPML 2 // No Termination (all fields of each cell are treated as CPML's)

//CPML Cell array: On device cell array containing all the necesarry parameters to do a CPML update.

#define NP_PF_CPML_CC 2 // psin-1 psin b c kappa
#define POS_POS_CPML_CC 0 // y=1 z=2
//--per field--
#define POS_PSIN1_CPML_CC 0 // Psinf+1 on x f+1-->y
#define POS_PSIN2_CPML_CC 1 // Psinf+2 on x f+2-->z

//CPMLcell link to property array
// Since access to the CPML parameters could be needed at all times. We encode it in the property array.

#define PROPNUM_CPML 2
#define BIT_CPML 26
#define FIRST_BIT_CPML 0
#define SELECTOR_CPML  (((((unsigned  long) 1)<<(BIT_CPML))-1)<<FIRST_BIT_CPML) //selector for the index of the cpml in global memory.
#define BIT_ISCPML 6
#define FIRST_BIT_ISCPML BIT_CPML
#define SELECTOR_ISCPML  (((((unsigned  long) 1)<<(BIT_ISCPML))-1)<<FIRST_BIT_ISCPML) // selector for the cpmltype.

// CPMLConconstant

#define THICK_CPML_MAX 32 // maximum number of fields needed in one dimentions. =ncpmlcells*2
#define GET_INDEX(d,index) index+d*THICK_CPML_MAX // get the index in the kernel.
#define PROPNUM_CPML_INDEX 3
#define FIRST_BIT_CPMLINDEX 0 // where you start you CPML indexing
#define BIT_CPMLINDEX_FIELD 5 // maximum cpml width is 16 cells. 2^5/2=16
#define SELECTOR_CPMLINDEX ((((unsigned  long) 1)<<(BIT_CPMLINDEX_FIELD))-1)<<FIRST_BIT_CPMLINDEX

//-----------Perfectly Conducting Layers------
#define NPPERFECTLAYER 7 // Number of parameters needed to define a perfect conducting layer

//----------On Device Parameter Array------------ // On the device we define a parameter array with a bunch of stuff we
// need to make the field update.

#define MAX_PARAM_NEEDED 6 //Maximum amount of parameters potentially needed to make a field update
#define POS_ABS_SOURCE_D 1 // Index of the first real amplitude Ex
#define POS_PHASE_SOURCE_D 2// Index of the first complex amplitude Ex
#define POS_OMEGA_SOURCE_D 3   // Index of the angular frequency of the source
#define POS_MUT_SOURCE_D 4  //Index of timestep of the gaussian envelope.
#define POS_SIGMAT_SOURCE_D 5 //Index of the temporal sigma of the gaussian envelope.
#define POS_MU_D 0// Place where we put the Mu for the magnetic field


 __constant__ float C1[1<<BIT_AMAT_FIELD];
 __constant__ float C2[1<<BIT_AMAT_FIELD];
 __constant__ float cgridx[SIMUL_SIZE_X],cgridy[SIMUL_SIZE_Y],cgridz[SIMUL_SIZE_Z]; // holds dx dy dz, for other purposes.
 __constant__ float cgridEx[SIMUL_SIZE_X],cgridEy[SIMUL_SIZE_Y],cgridEz[SIMUL_SIZE_Z]; // holds dt/dx,dt/dy,dt/dz used to update the Efield
 __constant__ float cgridHx[SIMUL_SIZE_X],cgridHy[SIMUL_SIZE_Y],cgridHz[SIMUL_SIZE_Z]; // holds dt/(dxmu) dt/(dymu) dt/(dzmu) to update the Hfield. Will be rescaled later to accomodate nonuniformgrid.
 __constant__ float clorentzreduced[(NP_FIXED_LORENTZ_C+NP_PP_LORENTZ_C*N_POLES_MAX)*NLORENTZ];// holds the constant material constants of lorentz materials in constant memory.
 __constant__ float kappa[THICK_CPML_MAX*3];// holds kappa in each direction at half yee cell intervals. First element is the start of the PML.
 __constant__ float bcpml[THICK_CPML_MAX*3]; // idem for b (7.102)
 __constant__ float ccpml[THICK_CPML_MAX*3]; // idem for c (7.99)
 __constant__ float cdt;
 __constant__ float *cfield[6];
 __constant__ int cxx,cyy,czz,ctt,ncpmlcells,nlorentzfields;
 __constant__ float *csource;
 __constant__ float *clorentzfield;
 __constant__ float **clorentzfieldaddress;
 __constant__ float *ccpmlcell;
 __constant__ unsigned long *cproperty[NUM_PROP_TOT];
 __constant__ unsigned long selectormattyoe=SELECTOR_MATTYPE;
 __constant__ unsigned long selectorsource=SELECTOR_SOURCE;
 __constant__ unsigned long selectordeps=SELECTOR_DEPS;
 __constant__ unsigned long selectorlorentz=SELECTOR_LORENTZ;
 __constant__ unsigned long selectorcpml=SELECTOR_CPML;
 __constant__ unsigned long selectoriscpml=SELECTOR_ISCPML;
 __constant__ unsigned long selectormattype[6]={SELECTOR_MATTYPE,SELECTOR_MATTYPE<<1*BIT_MATTYPE_FIELD,SELECTOR_MATTYPE<<2*BIT_MATTYPE_FIELD
 ,SELECTOR_MATTYPE<<3*BIT_MATTYPE_FIELD,SELECTOR_MATTYPE<<4*BIT_MATTYPE_FIELD,SELECTOR_MATTYPE<<5*BIT_MATTYPE_FIELD};
 __constant__ unsigned long selectorcpmlindex[6]={SELECTOR_CPMLINDEX,SELECTOR_CPMLINDEX<<1*BIT_CPMLINDEX_FIELD,SELECTOR_CPMLINDEX<<2*BIT_CPMLINDEX_FIELD
 ,SELECTOR_CPMLINDEX<<3*BIT_CPMLINDEX_FIELD,SELECTOR_CPMLINDEX<<4*BIT_CPMLINDEX_FIELD,SELECTOR_CPMLINDEX<<5*BIT_CPMLINDEX_FIELD};
 __constant__ unsigned long selectoramat[3]={SELECTOR_AMAT,SELECTOR_AMAT<<1*BIT_AMAT_FIELD,SELECTOR_AMAT<<2*BIT_AMAT_FIELD};
 __constant__ unsigned long selector_allfield_source[2]={SELECTOR_ALLFIELD_SOURCE_E,SELECTOR_ALLFIELD_SOURCE_H};
 __constant__ unsigned long selector_allfield_cpml[2]={SELECTOR_ALLFIELD_CPML_E,SELECTOR_ALLFIELD_CPML_H};


#endif	/* CONSTANTINIT */

