#include "header.h"
#include "headerRandomGPU.cu"
#include <iostream>
using namespace std;

#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18

typedef unsigned int uint32_t;
#define UINT32_C(a) ((uint32_t)a)

typedef struct {
    uint32_t aaa;
    int mm,nn,rr,ww;
    uint32_t wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint32_t maskB, maskC;
    int i;
    uint32_t *state;
}mt_struct;

typedef struct{
    unsigned int matrix_a;
    unsigned int mask_b;
    unsigned int mask_c;
    unsigned int seed;
} mt_struct_stripped;



//const int MT_RNG_COUNT = 4096;
static mt_struct MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];


__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];

//#define     DCMT_SEED 4172
//#define MT_RNG_PERIOD 607

__global__ void RandomGPU(double *d_Random, int NPerRng);
__device__ void BoxMuller(double& u1, double& u2);
__global__ void BoxMullerGPU(double *d_Random, int NPerRng);


extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//floor(a / b)
extern "C" int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
extern "C" int iAlignDown(int a, int b){
    return a - a % b;
}

extern "C" void initMTRef(const char *fname){
    
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTRef(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }

    for (int i = 0; i < MT_RNG_COUNT; i++){
        //Inline structure size for compatibility,
        //since pointer types are 8-byte on 64-bit systems (unused *state variable)
        if( !fread(MT + i, 16 /* sizeof(mt_struct) */ * sizeof(int), 1, fd) ){
            printf("initMTRef(): failed to load %s\n", fname);
            printf("TEST FAILED\n");
            exit(0);
        }
    }

    fclose(fd);
}

//Load twister configurations
void loadMTGPU(const char *fname){
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTGPU(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
        printf("initMTGPU(): failed to load %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    fclose(fd);
}

//Initialize/seed twister for current GPU context
void seedMTGPU(unsigned int seed){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
    cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT));
    //CUDA_SAFE_CALL( cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT)) );

    free(MT);
}

__global__ void RandomGPU(
    double *d_Random,
    int NPerRng
){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN];

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N){
        //Load bit-vector Mersenne Twister parameters
        mt_struct_stripped config = ds_MT[iRng];

        //Initialize current state
        mt[0] = config.seed;
        for(iState = 1; iState < MT_NN; iState++)
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

        iState = 0;
        mti1 = mt[0];
        for(iOut = 0; iOut < NPerRng; iOut++){
            //iState1 = (iState +     1) % MT_NN
            //iStateM = (iState + MT_MM) % MT_NN
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN) iState1 -= MT_NN;
            if(iStateM >= MT_NN) iStateM -= MT_NN;
            mti  = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
            mt[iState] = x;
            iState = iState1;

            //Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & config.mask_b;
            x ^= (x << MT_SHIFTC) & config.mask_c;
            x ^= (x >> MT_SHIFT1);

            //Convert to (0, 1] double and write to global memory
            d_Random[iRng + iOut * MT_RNG_COUNT] = ((double)x + 1.0) / 4294967296.0;
	    
        }
	//config.seed = mt[iState];
	//ds_MT[iRng] = config;
	ds_MT[iRng].seed = mt[iState];
    }
}



////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of NPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// NPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979
__device__ void BoxMuller(double& u1, double& u2){
    double   r = sqrtf(-2.0 * logf(u1));
    double phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void BoxMullerGPU(double *d_Random, int NPerRng){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N)
        for(int iOut = 0; iOut < NPerRng; iOut += 2)
            BoxMuller(
                d_Random[iRng + (iOut + 0) * MT_RNG_COUNT],
                d_Random[iRng + (iOut + 1) * MT_RNG_COUNT]
            );
}


void init_random_gpu(int SEED){
  const int PATH_N = 24000000;

  int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
  int RAND_N = MT_RNG_COUNT * N_PER_RNG;

  cudaMalloc((void**)&d_rand,RAND_N*sizeof(double));

  //const char *raw_path = cutFindFilePath("MersenneTwister.raw", argv[0]);
  //const char *dat_path = cutFindFilePath("MersenneTwister.dat", argv[0]);
  initMTRef("MersenneTwister.raw");
  loadMTGPU("MersenneTwister.dat");
  seedMTGPU(SEED);

  //RandomGPU<<<32, 128>>>(d_rand, N_PER_RNG);

  //BoxMullerGPU<<<32, 128>>>(d_rand, N_PER_RNG);

  //cudaFree(d_rand);
}

void free_random_gpu(){
  cudaFree(d_rand);
}





bool initializeRandomGPU(){

  if(thermostat==1){
    init_random_gpu(seed);
    cout << "INITIALIZE RANDOM NUMBER GPU : DONE" << endl;
    return 1;
  }
  else{
    return 1;
  }


}
