// Filename: errorBarsS2d.cpp
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
// errorBarsS2d numberFiles numberModes file1 file2 [file3 ...] > outputfile

// errorBarsS2d reads the radial structure factor generate with
// s2d and writes the mean value and the standard deviation
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
  ifstream fileinput[numberFiles];

  //Open files
  for(int i=0;i<numberFiles;i++){
    fileinput[i].open(argv[i+3]);
  }

  int periodic = 0;
  double kx, s, sm, error;
  cout.precision(15);

  for(int i=0;i<n;i++){
    sm = 0;
    error = 0;
    for(int j=0;j<numberFiles;j++){
      fileinput[j] >> kx >> s;
      
      s = s / (3.1129e-22 * ( 1 + 5.404895e+10/(pow(kx,4) + (1-periodic)*(24.6*pow(kx,2) + pow(4.73,4)))));
      
      error += j * (s-sm)*(s-sm)/double(j+1);
      sm += (s-sm)/double(j+1);
    }
    cout << kx << "    " << sm << "    " << sqrt(error/double(numberFiles)) << endl;
  }

  //Close files
  for(int i=0;i<numberFiles;i++){
    fileinput[i].close();
  }
  
}
