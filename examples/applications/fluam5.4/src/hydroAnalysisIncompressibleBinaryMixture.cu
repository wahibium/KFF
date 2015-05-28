// Filename: hydroAnalysisIncompressibleBinaryMixture.cu
//
// Copyright (c) 2010-2013, Florencio Balboa Usabiaga
//
// This file is part of Fluam
//
// Fluam is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Fluam is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Fluam. If not, see <http://www.gnu.org/licenses/>.



//#define GLOBALS_HYDROANALYSIS 1

#include "header.h"
#include "cells.h"
#include "fluid.h"
#include "headerOtherFluidVariables.h"
//Includes from TestHydroGrid.c
extern "C" {
#include "HydroGrid.h"
}
//#include "RNGs.h"
#define NDIMS 3
#include "hydroAnalysis.h"

bool hydroAnalysisIncompressibleBinaryMixture(int counter){

  if(counter == 0){
    ifstream fileinput;
    fileinput.open("hydroGridOptions.nml");
    string word, wordfile;
    while(!fileinput.eof()){
      getline(fileinput,word);
      wordfile += word + "\n";
    }
    fileinput.close();
    string fileOutName;
    fileOutName = outputname + ".hydroGridOptions.nml";
    ofstream fileout(fileOutName.c_str());
    fileout << wordfile << endl;
    fileout.close();

    //
    nCells[0] = mx;
    nCells[1] = my;
    nCells[2] = mz;
    systemLength[0] = lx;
    systemLength[1] = ly;
    systemLength[2] = lz;
    heatCapacity[0] = 1.;
    velocities = new double [NDIMS*mx*my*mz];
    //concent = new double [mx*my*mz];

    int project2D = 1 ; // Set to 1 if you want to do a 2D projection
    //cout << "CALLING createHydroAnalysis " << step << endl;
    string fileInputName = outputname + ".nml";
    //setHydroInputFile_C(fileInputName.c_str());
    
    //double scaling = densfluid/temperature; // This is not a good idea when temperature=0
    double scaling = 1;
    createHydroAnalysis_C(nCells,2,NDIMS,1,systemLength,heatCapacity,dt*samplefreq,0,scaling,project2D);
  }
  else if(counter == 1){
    for(int i=0;i<ncells;i++) {
      velocities[i] = cvx[i];
      velocities[i+ncells] = cvy[i];
      velocities[i+2*ncells] = cvz[i];
    }
    //cout << "CALLING updateHydroAnalysisMixture " << step << endl;
    //updateHydroAnalysisIsothermal_C(velocities, cDensity);
    updateHydroAnalysisMixture_C(velocities, cDensity, c);
  }
  else if(counter == 2){
    if((samplefreq>0) && (savefreq<=0)) {
      if(savefreq>=0) writeHydroGridMixture_C (cDensity, c, "", -1);
      writeToFiles_C(-1); // Write final statistics to files
    }  
    destroyHydroAnalysis_C();
    delete[] velocities;
    //delete[] concent;
  }
  else if(counter == 3){
    if(step==0){
      if(savefreq>=0) writeHydroGridMixture_C (cDensity, c, "", 0);
      writeToFiles_C(0); // Write to files
    }
    else{
      if(savefreq>=0) writeHydroGridMixture_C (cDensity, c, "", step/abs(savefreq));
      writeToFiles_C(step/abs(savefreq)); // Write to files
    }
  }
  else if(counter == 4){
    //cout << "CALLING projectHydroGrid "  << step << endl;
    // The value save_snapshot=2 means that a 2D projection will also be analyzed and saved to file
    // This requires project_2D=1 though and will overwrite any grid_2D data!
    //projectHydroGrid_C (cDensity, c, "", step, 2); // Number by time step
    // This will write both a full grid file and project along 2D:
    projectHydroGrid_C (cDensity, c, "", step/abs(savefreq), 2); // Number by snapshot
    // This will project along 2D but skip the 3D output:
    //projectHydroGrid_C (cDensity, c, "", step/abs(savefreq), -2);
  }
  return 1;
}



