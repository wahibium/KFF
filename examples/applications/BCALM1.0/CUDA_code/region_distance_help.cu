
#ifndef REGION_DISTANCE_HELP
#define REGION_DISTANCE_HELP

#define E_X 0
#define E_Y 1
#define E_Z 2
#define H_X 3
#define H_Y 4
#define H_Z 5
#define CENTER 6
#define POSITIVE_BORDER 7
#define NEGATIVE_BORDER 8

typedef struct region {
    int x, y, z; // Startting point of a region
    int dx, dy, dz; //Size of the region
    int type; // Type of the region.



} region;

void get_distance_yee(my_grid *g, int x0, int y0, int z0, int dx, int dy, int dz, int componentstart, int componentstop, float*ans, int dir) {
    //*!!!!! This is the conceptual yee cell for some reason the CPML dont work with the other cell and I was to lazy to find the bug.
    // In your head the comented yee cell gives you the right cell.

    float yee [9][3] = {
        {0.5, 0, 0}, //Ex //distances of the fields to the centerpoint of the cell in units of gridsize(x,y,z)
        {0, 0.5, 0}, //Ey
        {0, 0, 0.5}, //Ez
        {0, 0.5, 0.5}, //Hx
        {0.5, 0, 0.5}, //Hy
        {0.5, 0.5, 0}, //Hz
        {0, 0, 0}, //Center
        {0.5, 0.5, 0.5}, //Positive border
        {-0.5, -0.5, -0.5} //Negative border
    };

    /*  float yee [9][3] = {
            {0, 0.5, 0.5}, //Ex //distances of the fields to the centerpoint of the cell in units of gridsize(x,y,z)
            {0.5, 0, 0.5}, //Ey
            {0.5, 0.5, 0}, //Ez
            {0.5, 0, 0}, //Hx
            {0, 0.5, 0}, //Hy
            {0, 0, 0.5}, //Hz
            {0, 0, 0},//Center
            {1, 1, 1},//Border Next cell basically
            {0.5, 0.5, 0.5}//Middle
        }; // Center of the cell
     */


    // step one calculate the distance between the two cell centers.
    // dir is the direction.

    *ans = 0;

    switch (dir) {
        case 0:
            for (int i = x0; i < x0 + dx; i++) {
                *ans = *ans + g->gridX[i];
            }

            *ans = *ans - yee[componentstart][0] * g->gridX[x0] + yee[componentstop][0] * g->gridX[x0 + dx];
            //printf("\nansx=%f",*ansx);
            // if (*ansx==0) printf("\nyeestart=%d,yeestop=%d\n",componentstart,componentstop);

            break;

        case 1:
            for (int j = y0; j < y0 + dy; j++) {
                *ans = *ans + g->gridY[j];
            }
            *ans = *ans - yee[componentstart][1] * g->gridY[y0] + yee[componentstop][1] * g->gridY[y0 + dy];

            break;

        case 2:
            for (int k = z0; k < z0 + dz; k++) {
                *ans = *ans + g->gridZ[k];
            }
            *ans = *ans - yee[componentstart][2] * g->gridZ[z0] + yee[componentstop][2] * g->gridZ[z0 + dz];
            break;



    }


}

int InZone(my_grid* g, region R, int x, int y, int z)// Checks if the point x,y,z,field is in the zone.
 {
    float pos[3], center[3], diam[3]; // Help variables

    if (R.type != BOX) {


        for (int dir = 0; dir < 3; dir++) {

            get_distance_yee(g, R.x, R.y, R.z, R.dx, R.dy, R.dz, CENTER, CENTER, &diam[dir], dir); // Get the diameter in each direction
            get_distance_yee(g, R.x, R.y, R.z, x - R.x, y - R.y, z - R.z, CENTER, CENTER, &pos[dir], dir); // Get the position in each direction relative to the center of the circle
            center[dir] = diam[dir] / 2;
            pos[dir] = pos[dir] - center[dir];

        }
    }
    //printf("\npos=%e,center=%e",pos[0],center[0]);


    switch (R.type) {
            case BOX:
            if ((x <= (R.x + R.dx)) && (y <= (R.y + R.dy)) && (z <= (R.z + R.dz))) {
                return 1;
            } else {
                printf("An error has accured when assiging a zone at index x=%d y=%d z=%d", x, y, z);
                return 0;
            }
            break;






        case CYLINDER_X:

            if ((pow(pos[1], 2) + pow(pos[2], 2))<(pow(center[1], 2) + pow(center[2], 2)) / 2) {
                return 1;
            } else {
                return 0;
            }
        case CYLINDER_Y:

            if ((pow(pos[0], 2) + pow(pos[2], 2))<(pow(center[0], 2) + pow(center[2], 2)) / 2)
                return 1;
            else
                return 0;


        case CYLINDER_Z:


            if ((pow(pos[1], 2) + pow(pos[0], 2))<(pow(center[1], 2) + pow(center[0], 2)) / 2)
                return 1;
            else { //printf("\n pos: %e, diam: %e",(pow(pos[1], 2)+pow(pos[0], 2)),(pow(center[1], 2)+pow(center[0], 2)));
                return 0;
            }



        default:
            return 0;





    }

}





#endif
