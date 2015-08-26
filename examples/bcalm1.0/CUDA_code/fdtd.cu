#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include "DebugOutput.h"
#include "PrintLog.h"
#include "grid.cu"
#include "transfer.cu"
#include "kernel.cu"
#include <time.h>
#include <sys/times.h>
#include "ConstantMemoryInit.h"
#include "cutil_temp.h"
#include "populate.cu"
#include "bitoperations.cu"
#include "outs.cu"


// This file was around. I used it because nvcc could not find functions like CUDA_SAFE_CALL, presumebly present in cutil.h
// #include "memload_kernels.cu"

// Launch with -h as option to see usage

typedef struct options {
    char *infile, *outdir;
    int outputfrequency, fft, outtime;
} options;

void get_options(options *o, int argc, char *argv[]) {

    o->infile = NULL;
    o->outdir = NULL;
    o->outputfrequency = 2000; // default value
    o->fft = 0; // default no fft outputted
    o->outtime = 0; // default no timesteps outputted

    int c;
    int errflg = 0;
    extern char *optarg;
    extern int optind, optopt;

    while ((c = getopt(argc, argv, ":dhf:i:o:FT")) != -1)
        switch (c) {
            case 'd':
                break;
            case 'h':
                errflg++;
                break;
            case 'f':
                o->outputfrequency = atoi(optarg);
                break;
            case 'i':
                o->infile = optarg;
                break;
            case 'o':
                o->outdir = optarg;
                break;
            case 'F':
                o->fft = 1;
                break;
            case 'T':
                o->outtime = 1;
                break;
            case ':': /* -i or -o without operand */
                printf("Option -%c requires an operand\n", optopt);
                errflg++;
                break;
            case '?':
                printf("Unrecognized option: -%c\n", optopt);
                errflg++;
                break;



        }

    if (errflg) {
        printf(" \n\
usage:			\n\
	-i	Input File (default is InputFile) \n\
	-o	Output File (default is OutputFile) \n\
	-f	Progress frequency updates (in time steps) \n\
	-h	Prints this help message \n\
        -F      Generate FFT's \n\
        -T      Generate Timesteps \n\
                \n\
Example: \n\
	fdtd -i Inputfile -o Outputfile -f 1500 \n\
	fdtd -f 1500 -i Inputfile \n\
");
        exit(2);
    }

    // Set default values (if no option selected)
    if (!o->infile) {
        o->infile = new char[20];
        strcpy(o->infile, "InputFile");
    }
    if (!o->outdir) {
        o->outdir = new char[20];
        strcpy(o->outdir, "");
    }
}


FILE * simout; ///< Output log from the simulation
FILE * debugout;
char outdir[512];

int main(int argc, char *argv[]) {

    // Parse the command line
    options o; //structure with all the options from the command line
    get_options(&o, argc, argv);
    strcpy(outdir, o.outdir);

    // Simulation log
    char outfile[512];
    sprintf(outfile, "%s%s", outdir, SIMOUTFILE);
    simout = fopen(outfile, "w");
    sprintf(outfile, "%s%s", outdir, DEBUGOUTFILE);
    debugout = fopen(outfile, "w");

    my_grid g, d_g; // grid structures, d_g is the grid structure on the device

    // Debug output variables
    float* DEBUG_OUTPUT_D = debug_init_memory(); // init memory pointers


    // Time keeping variables
    int cntT; // current simulation step
    double dif, progress, eta, rate, nbcubesfactor;
    time_t start, now, last; // For keeping track of real time
    time(&start);

    // Read input file
    PrintLog("Initialize grid\n");
    grid_initialize(&g, o.infile);
    d_g = grid2device_field(g); // Intitialize the fields array(We need the addresses on the card).
    // Populate the grid
    populategrid(&g, &d_g);
    PrintLog("Initialize ouputs\n");
    out_zone* outarray = (out_zone*) malloc(sizeof (out_zone) * g.zozo);
    intialize_outzones(o.infile, &g, outarray);
    initialize_outHDF5(o.fft, o.outtime, outarray, &g);

    PrintLog("Transfer data to device\n"); // copy over data to device
    d_g = grid2device_rest(g, d_g);
    PrintLog("Transfer data to Constant Memory\n"); // copy over data to device
    Copy2ConstantMem(g);
    Copy2ConstantMemD(g, d_g);
    FreeLocalGrid(&g);

    // help define how the simulation will run on the GPU
    dim3 threads(B_XX, B_YY);
    dim3 grid(g.xx / (B_XX), g.yy / (B_YY));
    int threadslorentz = (B_XX_LORENTZ);
    int gridlorentz = g.nlorentzfields / (B_XX_LORENTZ);
    printf("Number of lorentzfields: %d\n", g.nlorentzfields);
    PrintLog("\nThread block dimensions: [%d, %d]\n", B_XX, B_YY);
    PrintLog("Grid dimensions: [%d, %d]\n", g.xx / B_XX, g.yy / B_YY);
    PrintLog("World dimensions: %d x %d x %d\n", g.xx, g.yy, g.zz);
    PrintLog("Simulation progress every %d time steps\n", o.outputfrequency);
    nbcubesfactor = (float) g.xx * (float) g.yy * (float) g.zz * (float) o.outputfrequency;

    PrintLog("***Starting simulation***\n");


// ----- Simultion----


    for (cntT = 0; cntT < g.tt; cntT++) {

        kernel_H << <grid, threads >> > (cntT); // update H-fields
        kernel_E << <grid, threads >> > (cntT); // update E-fields
        if (g.zlzl > 0)
            kernel_lorentz << <gridlorentz, threadslorentz >> >(); // update the Lorentz cells;

        // DEBUG OUTPUT
        PrintLogFile("DEBUG Extract = %e\n", debug_print(DEBUG_OUTPUT_D));

        fprintf(debugout, "%e\t%d\n", debug_print(DEBUG_OUTPUT_D), cntT);
        debug_reset(DEBUG_OUTPUT_D);

        export_outzones(outarray, &g, &d_g, cntT);

        if (cntT % o.outputfrequency == 0) {
            last = now;
            time(&now);
            dif = difftime(now, start); // Elapsed time in seconds
            dif = dif / 60; // Conversion to minutes
            rate = 60 * ((double) nbcubesfactor) / difftime(now, last);
            progress = ((float) cntT) / g.tt;
            eta = dif / progress - dif;
            PrintLog("%.2f : Step %d after %.2f minutes ETA:%.2f RATE:%.2e cubes / min\n",
                    progress * 100,
                    cntT, dif, eta, rate);
        }
    }


    // -----PostProcessing-----


    time(&now);
    dif = difftime(now, start); // Elapsed time in seconds
    rate = 60 * nbcubesfactor * g.tt / dif / (float) o.outputfrequency;
    PrintLog("100 : Kernel Terminated: after %.2f minutes. OVERALL RATE:%.2e cubes / min\n",
            100, dif / 60, rate);

    PrintLog("Free Device Memory\n");
    FreeCudaGrid(&d_g, g);

    PrintLog("\nStore result in hdf  file\n---\n");


    if (o.outtime == 1)
        write_temporal(outarray, &g, o.outdir);
    else
        PrintLog("No timesteps dumped\n");

    time(&start);
    if (o.fft == 1)
        write_fft(outarray, &g, o.outdir);
    else
        PrintLog("No fft's dumped\n");
    time(&now);
    dif = difftime(now, start); // Elapsed for FFT's
    printf("The FFT's took %f seconds",dif);


    PrintLog("Simulation terminated\n");
    // CUT_EXIT(argc, argv);

 



    return 0;
}
