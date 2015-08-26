#ifndef POVRAY_HELP_H
#define POVRAY_HELP_H

#include  "grid.cu"
#include "povray_help.cu"
#include "grid.cu"
#define MAX_COLORS 20 //maximum number of colors used


typedef struct mat_prop {

    float tranparency; //tranparancy
    float color[3]; // rgbcolors for it
    float phong;// brightness.

}mat_prop;

typedef struct sight_info{  // sets the camera and light sources and all.
float max_x,max_y,max_z;// the maximum of the of the latest object.

}sight_info;




typedef struct my_box {

    float xs,ys,zs,xe,ye,ze; // coordinates
    mat_prop prop;
 
}my_box;



void InitFile(FILE* file)

{fprintf(file,"#include \"colors.inc\" \n");
 fprintf(file,"#include \"woods.inc\" \n");
 fprintf(file,"#include \"stones.inc\" \n");
 fprintf(file,"#include \"golds.inc\" \n");
 fprintf(file,"#include \"glass.inc\" \n");
 
 fprintf(file,"background { color White*2 }");

}

void SetCamera(FILE* file,sight_info info)

{float cam_loc_x,cam_loc_y,cam_loc_z;

cam_loc_x=info.max_x/2;
cam_loc_y=info.max_y/2;
cam_loc_z=info.max_z*2;


float maxi=sqrt(pow(info.max_x/2,2)+pow(info.max_y/2,2));
float angle = 2*atan(maxi/cam_loc_z)*180/3.14;
angle=angle*2;// to be a bit aside



 fprintf(file,"camera {\n sky <0,0,1>\nlook_at  <%f, %f,  0>\nlocation<%f,%f,%f>\nup <0,0,1>\n right<-1,0,0>\n",cam_loc_x,cam_loc_y,cam_loc_x,cam_loc_y,cam_loc_z);
 fprintf(file,"angle %f\n",angle);
 fprintf(file,"}\n");
 fprintf(file,"light_source{\n <%f,%f,%f>\ncolor White*3\n ",cam_loc_x,cam_loc_y,cam_loc_z);
 fprintf(file,"}\n");
}





void  AddBox (FILE* file,my_box box)

{  fprintf(file,"\nbox{\n");
   fprintf(file,"<%f,%f,%f>,<%f,%f,%f>\n",box.xs,box.ys,box.zs,box.xe,box.ye,box.ze);
   fprintf(file,"texture {pigment { color rgb <%f, %f, %f> } }\n",box.prop.color[0],box.prop.color[1],box.prop.color[2]);
   fprintf(file,"}\n");  
}

void GetColor(mat_prop* prop,int index,int n) // Generates RGB colors

{
    prop->color[0]=0;
    prop->color[1]=0;
    prop->color[2]=0;

    for (int i=0;i<3;i++)
    {if( ( index % 3 ) == i)
    prop->color[i]=1-3*index/n;
    }





}


void CheckSight_Info(my_box box,sight_info* info) // Checks if the box is outside the Sightinfo if yes update sight info

{
    if (box.xe>info->max_x)
        info->max_x=box.xe;
    if (box.ye>info->max_y)
        info->max_y=box.ye;
    if (box.ze>info->max_z)
        info->max_z=box.ze;

}


void AddDielectricsPov(FILE* file,my_grid* g,sight_info* info)

{   int index; // Variables to help fill the array
    int x;// Variables to help fill the array
    int y;// Variables to help fill the array
    int z;// Variables to help fill the array
    int dx;// Variables to help fill the array
    int dy;// Variables to help fill the array
    int dz;// Variables to help fill the array
    my_box box;
    for (int zone=0;zone<g->zdzd;zone++)// don't do the first one

    {
    index=g->dielzone[zone*NPDIELZONE+0];
     x=g->dielzone[zone*NPDIELZONE+1];
     y=g->dielzone[zone*NPDIELZONE+2];
     z=g->dielzone[zone*NPDIELZONE+3];
     dx=g->dielzone[zone*NPDIELZONE+4];
     dy=g->dielzone[zone*NPDIELZONE+5];
     dz=g->dielzone[zone*NPDIELZONE+6];
     box.xs=x;
     box.ys=y;
     box.zs=z;
     box.xe=x+dx;
     box.ye=y+dy;
     box.ze=z+dz;
     box.prop.color[0]=index;
     box.prop.color[1]=0;
     box.prop.color[2]=0;
     GetColor(&box.prop,index,MAX_COLORS);
     CheckSight_Info(box,info);
     AddBox(file,box);
     }



}

void AddLorentzPov(FILE* file,my_grid* g,sight_info* info)

{

int index; // Variables to help fill the array
    int x;// Variables to help fill the array
    int y;// Variables to help fill the array
    int z;// Variables to help fill the array
    int dx;// Variables to help fill the array
    int dy;// Variables to help fill the array
    int dz;// Variables to help fill the array
     my_box box;
    for (int zone=0;zone<g->zlzl;zone++)

    {index=g->lorentzzone[zone*NPLORENTZZONE+0];
     x=g->lorentzzone[zone*NPLORENTZZONE+1];
     y=g->lorentzzone[zone*NPLORENTZZONE+2];
     z=g->lorentzzone[zone*NPLORENTZZONE+3];
     dx=g->lorentzzone[zone*NPLORENTZZONE+4];
     dy=g->lorentzzone[zone*NPLORENTZZONE+5];
     dz=g->lorentzzone[zone*NPLORENTZZONE+6];

     box.xs=x;
     box.ys=y;
     box.zs=z;
     box.xe=x+dx;
     box.ye=y+dy;
     box.ze=z+dz;
     GetColor(&box.prop,index+g->zdzd+1,MAX_COLORS);
     CheckSight_Info(box,info);
     AddBox(file,box);

    }


}








void AddAll (FILE* file,my_grid g,sight_info* info)

{ 
info->max_x=0;
info->max_y=0;
info->max_z=0;// Init the info.
    AddDielectricsPov(file,&g,info);    // Add the dielectrics.
    AddLorentzPov(file,&g,info);
    //printf("x %f y %f z %f",info->max_x,info->max_y,info->max_z);
}








#endif
