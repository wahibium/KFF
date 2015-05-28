// Filename: s2d.cpp
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
// s2d fileinput nx ny dx > outputfile
// with
// nx, ny = grid dimensions
// dx = meshwidth = h/my

// s2d reads the projection of
// the structure factor for a 2D system
// and writes in the standard output
// the radial structe factor
// k_discrete   S(k_discrete)
//
// with k_discrete = (2./dx) * sin(k * dx * 0.5)

#include <iostream>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>
using namespace std;


int main( int argc, char* argv[] ){
  
  string word;
  ifstream fileinput(argv[1]);
  int n = atoi(argv[2]);     //number of modes
  int ny = atoi(argv[3]);     //number of modes
  double dx = atof(argv[4]); //meshwidth dx=h/my
  
  double kx, s;

  getline(fileinput,word);
  getline(fileinput,word);
  
  cout.precision(15);

  double kdiscrete;
  for(int i=0;i<n;i++){
    fileinput >> kx >> s;
    kdiscrete = (2./dx) * fabs(sin(kx * dx * 0.5));
    if(kx>1e-6) // Only write the positive wavenumbers (negative are identically equal)
      cout << kdiscrete*(ny*dx) << " " << s << endl;
  }

  fileinput.close();

}
