// Filename: errorbarsS3d.cpp
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



// Use
// errorBarsS3d numberFiles numberWavevectors tolerance file1 file2 [file3 ...] > outputfile

// errorBarsS3d reads the radial structure factor generate with
// s3d and writes the mean value and the standard deviation
//
// kx  sMeanValue  sErrorBars
//
// IMPORTANT, all the input files should have the same modes.

#include <iostream>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>
#include <sstream>
using namespace std;


int main( int argc, char* argv[] ){
  string word;
  int numberFiles = atoi(argv[1]);
  int n = atoi(argv[2]);     //number of modes
  double tol = atof(argv[3]); // tolerance for grouping k's together
  ifstream fileinput[numberFiles];

  //Open files
  for(int i=0;i<numberFiles;i++){
    fileinput[i].open(argv[i+4]);
  }

  int periodic=1;
  double kx, kxold, s, sm, error;
  int count, bin;
  cout.precision(15);

  sm=0; error=0; count=0; kxold=0; bin=0;

  for(int i=0;i<n;i++){
    fileinput[0] >> kx >> s;    
    s = s / (3.1129e-22 * ( 1 + 5.404895e+10/(pow(kx,4) + (1-periodic)*(24.6*pow(kx,2) + pow(4.73,4)))));
        
    if(fabs(kx-kxold)<tol) { // Average over all wavenumbers close in magnitude
      error += count * (s-sm)*(s-sm)/double(count+1);
      sm += (s-sm)/double(count+1);
      count++;
      //cout << kxold << "    " << kx << "    " << bin << endl;
    }
    else {
      if (count>0) \
         cout << kxold << " " << sm << " " << sqrt(error/double(count)) << " " << count << endl;
      sm = 0;
      error = 0;
      count = 0;
      error += count * (s-sm)*(s-sm)/double(count+1);
      sm += (s-sm)/double(count+1);
      count++;
      bin++;
      kxold = kx;
      //cout << kxold << "    " << kx << "    " << bin << endl;      
    }

    for(int j=1;j<numberFiles;j++){
      fileinput[j] >> kx >> s;
      s = s / (3.1129e-22 * ( 1 + 5.404895e+10/(pow(kx,4) + (1-periodic)*(24.6*pow(kx,2) + pow(4.73,4)))));
      error += count * (s-sm)*(s-sm)/double(count+1);
      sm += (s-sm)/double(count+1);
      count++;
    }    
  }
  
  //Close files
  for(int i=0;i<numberFiles;i++){
    fileinput[i].close();
  }
  
}
