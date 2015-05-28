// Filename: radialKslice.cpp
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


#include <iostream>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>
#include <sstream>
using namespace std;


int main(  int argc, char* argv[]){

  double y;
  string word;
  ifstream fileinput(argv[1]);  //Input File
  string name = argv[2];        //Output Files
  int nSlices = atoi(argv[3]);  //Number of slices
  int n = atoi(argv[4]);        //Number of modes
  double dx = atof(argv[5]);    //Lowest mode
  double dy = dx;
  int elements = n*n;
  
  getline(fileinput,word);
  getline(fileinput,word);
  
  
  double kx[n], ky[n];
  double s[n*n];
  
  for(int l=0;l<nSlices;l++){
      fileinput >> y;
    for(int i=0;i<n;i++){
      kx[i] = dx * (i-n/2+1);
      ky[i] = dx * (i-n/2+1);
      //cout << kx[i] << " " << ky[i] << endl;
      for(int j=0;j<n;j++)
	fileinput >> s[i*n+j];
    }

    string nameOutput;
    stringstream ss;
    ss << l;
    nameOutput = name + "_" + ss.str() + ".dat";
    ofstream fileoutput(nameOutput.c_str());
    
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
	fileoutput << sqrt(ky[i]*ky[i]+kx[j]*kx[j]) << " " << s[i*n+j] << endl;
    }
    fileoutput.close();
  }

  
  fileinput.close();
  

}
