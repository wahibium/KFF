
#ifndef OUTS_H
#define OUTS_H
#include "grid.cu"
#include "hdf5.h"
#include "stdlib.h"
#include <unistd.h>
#include "fftw3.h"
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>


// Defines the start of an outzone.

typedef struct out_zone {
    int x[7], y[7], z[7]; // Startting point of a zone (1-6 for each field 7 for the size of the output)
    int dx[7], dy[7], dz[7]; //Size of a zone. (0-5 for each field 6 for the size of the output)
    int deltaT; //Time differential.
    char* name; // Name of the output.
    float fstart, fstop; // Frequenties we are interested in.
    int field[6]; //The fields we are going to output.
    float *dataT[6]; // Holds the temporal data for each field.
    float *datadumpT[6]; // Holds the extented zone timesteps to permit averaging before writing to the dataT;
    float *extendeddump; // Hold a larger zone to facilitate the averaging, if needed.
    float *dataF[6]; // Holds the data in the frequency domain.
    float *time; // Holds the real time in real units as data is sav
    int off_set_cnt; // Offset for the count.
    int step2save; // Amounts of timesteps to be saved.
    int freq2save; // Amount of frequencies computed during the fft.
    int indexfstart; // Index in the computed fft where we start exporting them.
    int freq2write; // Amout of frequencies written in the output.
    int stepsneeded; //Amout of elements on which we perform the fouriertranform
    int average; // Tells weather to average or not.
    float deltaF; //frequency differential in the fft output. ==1/(n*stepsneeded);
    float deltaFMin; //Min frequency resolution we want in our fourier transforms.
    hid_t file_idfft; // Output files are created before the simulation so we can free more memory
    hid_t file_idtime; // Output files are created before the simulation so we can free more memory
    // Zero padding will be performed to achieve that resolution;

} out_zone;

void set_zone_boundaries(my_grid *g, out_zone* zone) //If average in on the extra layers are gathered in order to do the correct averaging.
{
    int size;
    size = (zone->dx[6] + 1)*(zone->dy[6] + 1)*(zone->dz[6] + 1) * sizeof (float);
    // no averaging all the zones are exactly the same by default

    for (int f = 0; f < 6; f++) {
        zone->x[f] = zone->x[6];
        zone->y[f] = zone->y[6];
        zone->z[f] = zone->z[6];
        zone->dx[f] = zone->dx[6];
        zone->dy[f] = zone->dy[6];
        zone->dz[f] = zone->dz[6];

    }
    // By default the zone size is not increased



    if (zone->average == 1) { // Averaging in this zone
        if (zone->x[0] != 0) {
            zone->x[0] = zone->x[0] - 1; // For Ex get 1 Ex back
            zone->dx[0] = zone->dx[0] + 1;
            zone->x[4] = zone->x[0]; // For Hy get 1 Hy back
            zone->dx[4] = zone->dx[0];
            zone->x[5] = zone->x[0]; // For Hz get 1 Hz back
            zone->dx[5] = zone->dx[0];

        }

        if (zone->y[1] != 0) {
            zone->y[1] = zone->y[1] - 1; // For Ey get 1 Ey back
            zone->dy[1] = zone->dy[1] + 1;
            zone->y[3] = zone->y[0]; // For Hx get 1 Hx back
            zone->dy[3] = zone->dy[0];
            zone->y[5] = zone->y[0]; // For Hz get 1 Hz back
            zone->dy[5] = zone->dy[0];

        }

        if (zone->z[2] != 0) {
            zone->z[2] = zone->z[2] - 1; // For Ez get 1 Ez back
            zone->dz[2] = zone->dz[2] + 1;
            zone->z[3] = zone->z[0]; // For Hx get 1 Hx back
            zone->dz[3] = zone->dz[0];
            zone->z[4] = zone->z[0]; // For Hy get 1 Hy back
            zone->dz[4] = zone->dz[0];

        }
    }
}

void fillextendeddump(out_zone* zone, float* extendeddump, int f) // This function makes a larger zone to facilitate the averaging
{
    int idump, jdump, kdump, offset_exdump, offset_dump; // Where to fetch in the fields in timedump

    for (int i = 0; i <= zone->dx[6] + 1; i++) {
        idump = i; // If  zone->fulldir[0] = 1;
        if (zone->dx[f] == zone->dx[6] && i != 0) {
            idump = i - 1; // If zone->fulldir[0] and we are not on the border
        }
        for (int j = 0; j <= zone->dy[6] + 1; j++) {
            jdump = j; // If  zone->fulldir[0] = 1;

            if (zone->dy[f] == zone->dy[6] && j != 0) {
                jdump = j - 1; // If zone->fulldir[0] and we are not on the border
            }
            //  printf("\nbjdump%d",jdump);
            for (int k = 0; k <= zone->dz[6] + 1; k++) {
                kdump = k; // If  zone->fulldir[0] = 1;
                if (zone->dz[f] == zone->dz[6] && k != 0) {
                    kdump = k - 1; // If zone->fulldir[0] and we are not on the border
                }
                offset_exdump = i + (j)*(zone->dx[6] + 2)+
                        (k)*(zone->dy[6] + 2)*(zone->dx[6] + 2);
                offset_dump = idump + (jdump)*(zone->dx[f] + 1)+
                        (kdump)*(zone->dy[f] + 1)*(zone->dx[f] + 1);
                extendeddump[offset_exdump] = zone->datadumpT[f][offset_dump];


            }
        }
    }
}

void filloffsetedump(int i, int j, int k, int dx, int dy, int offset_edump[4], int f) {
    switch (f) {
        case 0: // Ex
            offset_edump[0] = i + (j + 1)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            offset_edump[1] = i + 1 + (j + 1)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            break;
        case 1: // Ey
            offset_edump[0] = i + 1 + (j)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            offset_edump[1] = i + 1 + (j + 1)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            break;
        case 2: // Ez
            offset_edump[0] = i + 1 + (j + 1)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[1] = i + 1 + (j + 1)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            break;
        case 3: // Hx
            offset_edump[0] = i + (j)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[1] = i + (j)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            offset_edump[2] = i + (j + 1)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[3] = i + (j + 1)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            break;
        case 4: // Hy
            offset_edump[0] = i + (j)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[1] = i + (j)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            offset_edump[2] = (i + 1) + (j)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[3] = (i + 1) + (j)*(dx + 2)+(k + 1)*(dy + 2)*(dx + 2);
            break;
        case 5: // Hz
            offset_edump[0] = i + (j)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[1] = i + (j + 1)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[2] = (i + 1) + (j + 1)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            offset_edump[3] = (i + 1) + (j)*(dx + 2)+(k)*(dy + 2)*(dx + 2);
            break;


    }
}

void get_average(out_zone* zone) // Calculate the average if needed. Otherwise just make the copy.
{
    int size, offset_output;
    int offset_edump[4];
    size = (zone->dx[6] + 1)*(zone->dy[6] + 1)*(zone->dz[6] + 1) * sizeof (float);


    // Just copy if no average
    if (zone->average == 0) {
        for (int f = 0; f < 6; f++) {
            if (zone->field[f] == 1) {
                memcpy(&zone->dataT[f][zone->off_set_cnt], &zone->datadumpT[f][0], size);

            };
        }
    }

    // When averaging is needed;
    if (zone->average == 1) {

        // Electrical Fields
        for (int f = 0; f < 3; f++) {
            if (zone->field[f] == 1) {
                fillextendeddump(zone, zone->extendeddump, f); // Fills an array bigger than the output in every dimention
                // cover the whole zone
                for (int i = 0; i <= zone->dx[6]; i++) {
                    for (int j = 0; j <= zone->dy[6]; j++) {
                        for (int k = 0; k <= zone->dz[6]; k++) {

                            offset_output = zone->off_set_cnt + i + (j)*(zone->dx[6] + 1)+
                                    (k)*(zone->dy[6] + 1)*(zone->dx[6] + 1);
                            filloffsetedump(i, j, k, zone->dx[6], zone->dy[6], offset_edump, f);
                            for (int cnt = 0; cnt < 2; cnt++)
                                zone->dataT[f][offset_output] = zone->dataT[f][offset_output] + zone->extendeddump[offset_edump[cnt]] / 2; // Averaging the H
                        }

                    }

                }

            }
        }


        // Magnetic fields
        for (int f = 3; f < 6; f++) {
            if (zone->field[f] == 1) {
                fillextendeddump(zone, zone->extendeddump, f); // Fills an array bigger than the output in every dimention
                // cover the whole zone
                for (int i = 0; i <= zone->dx[6]; i++) {
                    for (int j = 0; j <= zone->dy[6]; j++) {
                        for (int k = 0; k <= zone->dz[6]; k++) {

                            offset_output = zone->off_set_cnt + i + (j)*(zone->dx[6] + 1)+
                                    (k)*(zone->dy[6] + 1)*(zone->dx[6] + 1);
                            filloffsetedump(i, j, k, zone->dx[6], zone->dy[6], offset_edump, f);
                            for (int cnt = 0; cnt < 4; cnt++)
                                zone->dataT[f][offset_output] = zone->dataT[f][offset_output] + zone->extendeddump[offset_edump[cnt]] / 4; // Averaging the H
                        }

                    }

                }

            }
        }
    }


}

void get_property_cell(out_zone myzone, my_grid* g, hid_t file_id) {// create the array.

    /*     unsigned long *** type;
        type = (unsigned long***) malloc(sizeof (unsigned long**) *(myzone.dx + 1));
        for (int i = 0; i <= myzone.dx; i++) {
            type[i] = (unsigned long**) malloc(sizeof (unsigned long*) *(myzone.dy + 1));
            for (int j = 0; j <= myzone.dy; j++)
                type[i][j] = (unsigned long*) malloc(sizeof (unsigned long ) *(myzone.dz + 1));
        }*/

    unsigned long* type = (unsigned long*) malloc(sizeof (unsigned long**) *(myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1));
    unsigned long* debug = (unsigned long*) malloc(sizeof (unsigned long**) *(myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1));
    unsigned long* amat = (unsigned long*) malloc(sizeof (unsigned long**) *(myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1));

    for (int i = myzone.x[6]; i <= myzone.dx[6] + myzone.x[6]; i++) {
        for (int j = myzone.y[6]; j <= myzone.dy[6] + myzone.y[6]; j++) {
            for (int k = myzone.z[6]; k <= myzone.dz[6] + myzone.z[6]; k++) {

                type[(i - myzone.x[6])+(j - myzone.y[6])*(myzone.dx[6] + 1)+(k - myzone.z[6])*(myzone.dx[6] + 1)*(myzone.dy[6] + 1)] =           \
                        g->property[PROPNUM_DEPS][(i) + (j) * g->xx + (k) * g->xx * g->yy]; //contains everything I need
                debug[(i - myzone.x[6])+(j - myzone.y[6])*(myzone.dx[6] + 1)+(k - myzone.z[6])*(myzone.dx[6] + 1)*(myzone.dy[6] + 1)] =          \
 g->property[PROPNUM_CPML][(i) + (j) * g->xx + (k) * g->xx * g->yy]; //contains everything I need
                amat[(i - myzone.x[6])+(j - myzone.y[6])*(myzone.dx[6] + 1)+(k - myzone.z[6])*(myzone.dx[6] + 1)*(myzone.dy[6] + 1)] =          \
 g->property[PROPNUM_AMAT][(i) + (j) * g->xx + (k) * g->xx * g->yy]; //contains everything I need
            }
        }
    }
    hsize_t dims[3];

    dims[0] = myzone.dx[6] + 1;
    dims[1] = myzone.dy[6] + 1;
    dims[2] = myzone.dz[6] + 1;

    H5LTmake_dataset(file_id, "/property", 3, dims, H5T_NATIVE_ULONG, type);
    H5LTmake_dataset(file_id, "/amat", 3, dims, H5T_NATIVE_ULONG, amat);
    H5LTmake_dataset(file_id, "/debug", 3, dims, H5T_NATIVE_ULONG, debug);
}

void get_mattype_cell(out_zone myzone, my_grid* g, hid_t file_id) {// create the array.

    /*     unsigned long *** type;
        type = (unsigned long***) malloc(sizeof (unsigned long**) *(myzone.dx + 1));
        for (int i = 0; i <= myzone.dx; i++) {
            type[i] = (unsigned long**) malloc(sizeof (unsigned long*) *(myzone.dy + 1));
            for (int j = 0; j <= myzone.dy; j++)
                type[i][j] = (unsigned long*) malloc(sizeof (unsigned long ) *(myzone.dz + 1));
        }*/


    unsigned long* mattype = (unsigned long*) malloc(sizeof (unsigned long**) *(myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1));

    for (int i = myzone.x[6]; i <= myzone.dx[6] + myzone.x[6]; i++) {
        for (int j = myzone.y[6]; j <= myzone.dy[6] + myzone.y[6]; j++) {
            for (int k = myzone.z[6]; k <= myzone.dz[6] + myzone.z[6]; k++) {

                mattype[(i - myzone.x[6])+(j - myzone.y[6])*(myzone.dx[6] + 1)+(k - myzone.z[6])*(myzone.dx[6] + 1)*(myzone.dy[6] + 1)] = g->property[PROPNUM_MATTYPE][(i) + (j) * g->xx + (k) * g->xx * g->yy]; //contains everything I need


            }

        }


    }
    hsize_t dims[3];

    dims[0] = myzone.dx[6] + 1;
    dims[1] = myzone.dy[6] + 1;
    dims[2] = myzone.dz[6] + 1;

    H5LTmake_dataset(file_id, "/mattype", 3, dims, H5T_NATIVE_ULONG, mattype);
}

void write_attributes(out_zone zone, my_grid *g, hid_t file_id) // writes all the attributes.
{
    hsize_t dims[3];
    // Exporting the grids.
    dims[0] = g->xx;
    H5LTmake_dataset(file_id, "/gridX", 1, dims, H5T_NATIVE_FLOAT, g->gridX);
    dims[0] = g->yy;
    H5LTmake_dataset(file_id, "/gridY", 1, dims, H5T_NATIVE_FLOAT, g->gridY);
    dims[0] = g->zz;
    H5LTmake_dataset(file_id, "/gridZ", 1, dims, H5T_NATIVE_FLOAT, g->gridZ);
    //Exporting the start and the stop of the zones

    int start[3] = {zone.x[6], zone.y[6], zone.z[6]};
    int delta[3] = {zone.dx[6], zone.dy[6], zone.dz[6]};

    dims[0] = 3;
    H5LTmake_dataset(file_id, "/start", 1, dims, H5T_NATIVE_INT, start);
    dims[0] = 3;
    H5LTmake_dataset(file_id, "/delta", 1, dims, H5T_NATIVE_INT, delta);

    //Exporting the property array.
    get_property_cell(zone, g, file_id);
    //Exporting the mattype arrat.
    get_mattype_cell(zone, g, file_id);




}

void initialize_outHDF5(int fft, int outtime, out_zone* outarray, my_grid *g) {
    hid_t file_id;
    for (int zone = 0; zone < g->zozo; zone++) {

        if (outtime == 1) {
            char temp[MAX_CHAR_OUT];
            char* hdf5_filename = outarray[zone].name;
            printf("Creating hdf5 file %s...\n", hdf5_filename);
            sprintf(temp, "%s%s", outdir, hdf5_filename);
            //printf("%s\n", temp);
            file_id = H5Fcreate(temp, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // create the hdf5 file
            outarray[zone].file_idtime = file_id;
            printf("file id: %d\n", file_id);
            write_attributes(outarray[zone], g, file_id);

        }


        if (fft == 1) {
            char temp[MAX_CHAR_OUT];
            char* hdf5_filename = outarray[zone].name;
            sprintf(hdf5_filename, "%sFFT", hdf5_filename);
            printf("Creating hdf5 file %s...\n", hdf5_filename);
            sprintf(temp, "%s%s", outdir, hdf5_filename);
            file_id = H5Fcreate(temp, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // create the hdf5 file
            outarray[zone].file_idfft = file_id;
            printf("file id: %d\n", file_id);
            write_attributes(outarray[zone], g, file_id);

        }

    }


}

void intialize_outzones(char *hdf5_filename, my_grid *g, out_zone* outarray) {

    char pad[1] = {'?'};


    /* Handles */
    hid_t file_id; // file id for hdf file
    // temporary variables
    printf("Opening hdf5 file %s...", hdf5_filename);
    file_id = H5Fopen(hdf5_filename, H5F_ACC_RDONLY, H5P_DEFAULT); // open the hdf5 file
    if (file_id <= 0)
        error(-1, 0, "error.\n");
    printf("file id: %d\n", file_id);

    // The names are padded with question marks in one large array. I did not manage to transfer an array of strings genereted om matlab.

    float* zonearray = (float*) cust_alloc(sizeof (float) * g->zozo * NP_OUTPUTZONE);

    H5LTread_dataset(file_id, "/in/outputzones", H5T_NATIVE_FLOAT, zonearray);
    char** names = (char**) malloc(sizeof (char*) * g->zozo);
    char * temp = (char*) malloc(sizeof (char) * MAX_CHAR_OUT * g->zozo + 1);
    H5LTread_dataset_string(file_id, "/in/outputzonesname", temp);


    for (int cnt = 0; cnt < MAX_CHAR_OUT * g->zozo; cnt++) {
        if (!strncmp(&temp[cnt], &pad[0], 1)) {
            temp[cnt] = '\0'; //null pointer
        }
    }


    //allocate space for each string array.
    for (int zone = 0; zone < g->zozo; zone++) {
        names[zone] = (char*) malloc(sizeof (char) * MAX_CHAR_OUT + 1);
        strncpy(names[zone], &temp[zone * MAX_CHAR_OUT], MAX_CHAR_OUT);

    }


    for (int zone = 0; zone < g->zozo; zone++) {
        outarray[zone].x[6] = zonearray[zone * NP_OUTPUTZONE + POS_START_O];
        outarray[zone].y[6] = zonearray[zone * NP_OUTPUTZONE + POS_START_O + 1];
        outarray[zone].z[6] = zonearray[zone * NP_OUTPUTZONE + POS_START_O + 2];
        outarray[zone].dx[6] = zonearray[zone * NP_OUTPUTZONE + POS_DSTART_O];
        outarray[zone].dy[6] = zonearray[zone * NP_OUTPUTZONE + POS_DSTART_O + 1];
        outarray[zone].dz[6] = zonearray[zone * NP_OUTPUTZONE + POS_DSTART_O + 2];
        outarray[zone].deltaT = zonearray[zone * NP_OUTPUTZONE + POS_DELTAT_O];
        outarray[zone].fstart = zonearray[zone * NP_OUTPUTZONE + POS_FOUTSTART_O];
        outarray[zone].fstop = zonearray[zone * NP_OUTPUTZONE + POS_FOUTSTOP_O];
        outarray[zone].name = names[zone];
        outarray[zone].deltaFMin = zonearray[zone * NP_OUTPUTZONE + POS_DELTAFMIN_0];
        outarray[zone].average = zonearray[zone * NP_OUTPUTZONE + POS_AVERAGE];

        set_zone_boundaries(g, &outarray[zone]);


        //printf("\n DeltaFMin is equal to: %e", outarray[zone].deltaFMin);
        outarray[zone].step2save = floor(g->tt / outarray[zone].deltaT) + 1;
        // get how much padding we need.
        int stepsneeded = 1;
        do {
            stepsneeded = stepsneeded * 2;
            outarray[zone].deltaF = 1 / ((float) stepsneeded * g->dt * outarray[zone].deltaT);
        } while ((stepsneeded <= outarray[zone].step2save) || (outarray[zone].deltaF > outarray[zone].deltaFMin)); // We zero pad to have a exponential of 2 and enough frequency resolution.
        //printf("\nAverage=%d\n", outarray[zone].average);
        outarray[zone].stepsneeded = stepsneeded;
        outarray[zone].freq2save = floor(outarray[zone].stepsneeded / 2) + 1;

        // Looking for the index where we will start outputting the array.
        int indexfstart = 0;

        while ((indexfstart * outarray[zone].deltaF < outarray[zone].fstart) && (indexfstart < outarray[zone].freq2save - 1)) {
            indexfstart = indexfstart + 1;


        }

        outarray[zone].indexfstart = indexfstart;
        outarray[zone].fstart = outarray[zone].deltaF*indexfstart;
        //printf("\nindexstart=%d",indexfstart);
        //printf("\nfstart=%e",outarray[zone].fstart);
        // Looking for how long the outputfrequencyarray has to be.

        int indexfstop = 0;
        while ((indexfstop * outarray[zone].deltaF < outarray[zone].fstop) && (indexfstop < outarray[zone].freq2save - 1))
            indexfstop = indexfstop + 1;

        outarray[zone].freq2write = indexfstop - indexfstart + 1;
        outarray[zone].fstop = outarray[zone].deltaF*indexfstop;
        //printf("\nindexstart=%d",indexfstop);
        //printf("\nfstop=%e",outarray[zone].fstop);

        if (outarray[zone].freq2write < 1) {
            printf("\nError with the frequency-limits in the output of zone%s", outarray[zone].name);
            assert(0);
        }
        // Allocate space for the definite data;
        for (int f = 0; f < 6; f++) {
            outarray[zone].field[f] = zonearray[zone * NP_OUTPUTZONE + POS_FIELD_O + f];
            if (outarray[zone].field[f] == 1) {// allocate space for the data.
                //printf("deltaT[zone]=%d", outarray[zone].deltaT);
                outarray[zone].dataT[f] = (float*) malloc((outarray[zone].dx[6] + 1)*(outarray[zone].dz[6] + 1)*(outarray[zone].dy[6] + 1) * outarray[zone].step2save * sizeof (float));
            }
        }
        // Allocate space for the extended zones (just holds one timestep) before being copied into dataT[f]
        for (int f = 0; f < 6; f++) {

            if (outarray[zone].field[f] == 1) {// allocate space for the data.
                //printf("deltaT[zone]=%d", outarray[zone].deltaT);
                outarray[zone].datadumpT[f] = (float*) malloc((outarray[zone].dx[f] + 1)*(outarray[zone].dz[f] + 1)*(outarray[zone].dy[f] + 1) * sizeof (float))
                        ; // Just one timestep.
            }
        }
        outarray[zone].extendeddump = (float*) malloc((outarray[zone].dx[6] + 2)*(outarray[zone].dz[6] + 2)*(outarray[zone].dy[6] + 2) * sizeof (float)); // This dump assumes a larger zone so the postprocessing is eased

        //Allocate data for the time stamps im real units of time.
        outarray[zone].time = (float*) malloc(outarray[zone].step2save * sizeof (float));

        //Set offsetcount to zero

        outarray[zone].off_set_cnt = 0;


    }

}

void export_outzones(out_zone* outarray, my_grid* g_h, my_grid* d_g, int cnt) {

    int offset_output, offset_grid;
    for (int zone = 0; zone < d_g->zozo; zone++) // Go over all the zones.
    {
        if (cnt % outarray[zone].deltaT == 0) {
            //printf("cnt=%d,off_set_cnt=%d\n",cnt,outarray[zone].off_set_cnt);
            //printf("Cnt/deltaT=%d\n",cnt/outarray[zone].deltaT);

            for (int f = 0; f < 6; f++) {
                if (outarray[zone].field[f] == 1) {
                    for (int k = outarray[zone].z[f]; k <= outarray[zone].z[f] + outarray[zone].dz[f]; k++) {
                        for (int j = outarray[zone].y[f]; j <= outarray[zone].y[f] + outarray[zone].dy[f]; j++) {

                            offset_output = (j - outarray[zone].y[f])*(outarray[zone].dx[f] + 1)+(k - outarray[zone].z[f])*(outarray[zone].dy[f] + 1)*(outarray[zone].dx[f] + 1);
                            offset_grid = outarray[zone].x[f] + (j) * g_h->xx + k * g_h->xx * g_h->yy;
                            //printf("After ExportZones\n");



                            CUDA_SAFE_CALL(cudaMemcpy(&(outarray[zone].datadumpT[f][offset_output]), &(d_g->field[f][offset_grid]), sizeof (float) * (outarray[zone].dx[f] + 1), cudaMemcpyDeviceToHost));


                        }
                    }
                }
            }
            get_average(&outarray[zone]); // Do the averaging.
            outarray[zone].off_set_cnt = outarray[zone].off_set_cnt + (outarray[zone].dx[6] + 1)*(outarray[zone].dz[6] + 1)*(outarray[zone].dy[6] + 1);

        }
    }
}

void calculate_fft(out_zone zone, double*fftReal, double*fftImag, int f) {
    fftw_complex *out;
    fftw_plan p;
    double *in;
    // allocate space for the result of the ffttrace

    out = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * zone.stepsneeded+1);
    in = (double*) malloc(sizeof (double) * zone.stepsneeded); // Allocate the dataspace on which we operate the fourier transform
    memset(in, 0, sizeof (double) * zone.stepsneeded);

    // Appearantly creating a plan destroys the first transform
    p = fftw_plan_dft_r2c_1d(zone.stepsneeded, in, out, FFTW_PATIENT);

    for (int k = 0; k <= zone.dz[6]; k++) {
        for (int j = 0; j <= zone.dy[6]; j++) {
            for (int i = 0; i <= zone.dx[6]; i++) {
              
                // Copy the timetrace
                for (int t = 0; t < zone.step2save; t++) {

                    unsigned int offsetT = t * (zone.dx[6] + 1)*(zone.dz[6] + 1)*(zone.dy[6] + 1);
                    unsigned int offset_output = offsetT + i + (j)*(zone.dx[6] + 1)+(k)*(zone.dy[6] + 1)*(zone.dx[6] + 1);
                    unsigned int offsettime = t;
                    in[offsettime] = (double) zone.dataT[f][offset_output];
                }

                fftw_execute(p); /* repeat as needed */


                for (int freq = 0; freq < zone.freq2write; freq++) {


                    unsigned int offsetF = freq * (zone.dx[6] + 1)*(zone.dz[6] + 1)*(zone.dy[6] + 1);
                    unsigned int frequencyindex = offsetF + i + (j)*(zone.dx[6] + 1)+(k)*(zone.dy[6] + 1)*(zone.dx[6] + 1);
                   
                    fftReal[frequencyindex] = out[freq + zone.indexfstart][0] / zone.stepsneeded;
                    fftImag[frequencyindex] = out[freq + zone.indexfstart][1] / zone.stepsneeded;

                }

            }
        }
    }
    fftw_destroy_plan(p);
    fftw_free(out);
    free(in);



}




void write_fft(out_zone* outarray, my_grid*g, char *outdir) // Will write the fourier transforms of the timesteparran a new file.
{

    hid_t file_id;
    hsize_t dims[4];

    char *field_names[6] = {"/Ex", "/Ey", "/Ez", "/Hx", "/Hy", "/Hz"}; // names of the field components
    double*fftReal;
    double *fftImag;
    //***Make a timetracearray****//

    for (int zone = 0; zone < g->zozo; zone++) {

        // Allocate a time trace array. for each zone
        out_zone myzone = outarray[zone];

        file_id = myzone.file_idfft;
        fftReal = (double*) malloc(sizeof (double) * myzone.freq2write * (myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1)); // Stores the abosulute value.
        fftImag = (double*) malloc(sizeof (double) * myzone.freq2write * (myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1)); // Stores the phase.

        for (int f = 0; f < 6; f++) {
            if (myzone.field[f] == 1) {// for each field that we want to output.
                //populate the timesteparray.

                fftReal = (double*) memset(fftReal, 0, (sizeof (double) * myzone.freq2write * (myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1))); // Stores the abosulute value.
                fftImag = (double*) memset(fftImag, 0, (sizeof (double) * myzone.freq2write * (myzone.dx[6] + 1)*(myzone.dy[6] + 1)*(myzone.dz[6] + 1))); // Stores the phase.

                calculate_fft(myzone, fftReal, fftImag, f);

                //Write to HDF5
                char REAL[10], IMAG[10], ABS[10], PHASE[10];
                sprintf(REAL, "%s_real", field_names[f]);
                sprintf(IMAG, "%s_imag", field_names[f]);
                sprintf(ABS, "%s_abs", field_names[f]);
                sprintf(PHASE, "%s_phase", field_names[f]);
                // determine the dimensions for the write
                dims[0] = myzone.dx[6] + 1;
                dims[1] = myzone.dy[6] + 1;
                dims[2] = myzone.dz[6] + 1;
                dims[3] = myzone.freq2write;

                H5LTmake_dataset(file_id, REAL, 4, dims, H5T_NATIVE_DOUBLE, fftReal); // transfer to the hdf5 file
                H5LTmake_dataset(file_id, IMAG, 4, dims, H5T_NATIVE_DOUBLE, fftImag); // transfer to the hdf5 file

            }
        }


        dims[0] = 1;
        H5LTmake_dataset(file_id, "/deltaData", 1, dims, H5T_NATIVE_FLOAT, &myzone.deltaF); // DeltaF in Hz
        H5LTmake_dataset(file_id, "/startData", 1, dims, H5T_NATIVE_FLOAT, &myzone.fstart); // StartF in Hz
        H5LTmake_dataset(file_id, "/stopData", 1, dims, H5T_NATIVE_FLOAT, &myzone.fstop); // StopF in Hz
        char dataname[20] = {"frequency (Hz)"};
        H5LTmake_dataset_string(file_id, "/nameData", dataname);
        // Freeing Memory

        free(fftReal);
        free(fftImag);
        printf("Closing hdf5 file (file id: %d)...\n", file_id);
        H5Fclose(file_id);

    }


}

void write_temporal(out_zone* outarray, my_grid* g, char* outdir) {
    hid_t file_id;
    hsize_t dims[4];

    char *field_names[6] = {"/Ex", "/Ey", "/Ez", "/Hx", "/Hy", "/Hz"}; // names of the field components


    for (int zone = 0; zone < g->zozo; zone++) {
        int f;
        file_id = outarray[zone].file_idtime;

        // put this in so we know how many outs there are

        // determine the dimensions
        dims[0] = outarray[zone].dx[6] + 1;
        dims[1] = outarray[zone].dy[6] + 1;
        dims[2] = outarray[zone].dz[6] + 1;
        dims[3] = outarray[zone].step2save;







        for (f = 0; f < 6; f++) {
            if (outarray[zone].field[f] == 1) {
                //my_stringcat ("/out", cnt, field_names[f], temp);
                //printf("%s\n",temp);

                H5LTmake_dataset(file_id, field_names[f], 4, dims, H5T_NATIVE_FLOAT, outarray[zone].dataT[f]); // transfer to the hdf5 file
            }
        }

        float delta = outarray[zone].deltaT * g->dt;
        float start = 0;
        float stop = (outarray[zone].step2save - 1) * outarray[zone].deltaT * g->dt;
        //(g->tt-1)*g->dt;
        char dataname[20] = {"time (s)"};
        dims[0] = 1;
        H5LTmake_dataset(file_id, "/deltaData", 1, dims, H5T_NATIVE_FLOAT, &delta); // DeltaF in Hz
        H5LTmake_dataset(file_id, "/startData", 1, dims, H5T_NATIVE_FLOAT, &start); // StartF in Hz
        H5LTmake_dataset(file_id, "/stopData", 1, dims, H5T_NATIVE_FLOAT, &stop); // StopF in Hz
        H5LTmake_dataset_string(file_id, "/nameData", dataname);
        printf("Closing hdf5 file (file id: %d)...\n", file_id);
        H5Fclose(file_id);
    }
    return;
}
#endif
