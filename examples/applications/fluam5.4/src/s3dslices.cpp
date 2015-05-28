// Filename: s3dslices.cpp
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
// s3dslices inputfile outputfile numberOfSlices numberOfModes lowestMode meshwitdh

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
  double kmin = atof(argv[5]);    //Lowest mode
  double dx = atof(argv[6]); // meshwidth
  int elements = n*n;
  double kxAux;
  int m;
  
  getline(fileinput,word);
  getline(fileinput,word);
  
  
  double kx[n*n], ky[n*n], k[n*n];
  double s[n*n];
  
  for(int slice=0;slice<nSlices;slice++){
    fileinput >> y;
    for(int i=0;i<n;i++){
      //cout << kx[i] << " " << ky[i] << endl;
      for(int j=0;j<n;j++){
	kx[i*n+j] = kmin * (i-n/2+1);
	ky[i*n+j] = kmin * (j-n/2+1);
	k[i*n + j] = sqrt(kx[i*n + j]*kx[i*n + j] + ky[i*n + j]*ky[i*n + j]);
	fileinput >> s[i*n+j];
      }
    }

    int l = elements;
    while(l>1){
      m=0;
      for(int i=0;i<l-1;i++){
	if(k[i] > k[i+1]){
	  kxAux = kx[i]; kx[i] = kx[i+1]; kx[i+1] = kxAux;
	  kxAux = ky[i]; ky[i] = ky[i+1]; ky[i+1] = kxAux;
	  kxAux = s[i];  s[i] = s[i+1];   s[i+1] = kxAux;
	  kxAux = k[i];  k[i] = k[i+1];   k[i+1] = kxAux;
	  m = i + 1;
	}
	else if((fabs(k[i]-k[i+1])<=1e-6)&&(kx[i]>kx[i+1])){
	  kxAux = kx[i]; kx[i] = kx[i+1]; kx[i+1] = kxAux;
	  kxAux = ky[i]; ky[i] = ky[i+1]; ky[i+1] = kxAux;
	  kxAux = s[i];  s[i] = s[i+1];   s[i+1] = kxAux;
	  kxAux = k[i];  k[i] = k[i+1];   k[i+1] = kxAux;
	  m = i + 1;
	}
	else if((fabs(k[i]-k[i+1])<=1e-6)&&(fabs(kx[i]-kx[i+1])<=1e-6)&&(ky[i]>ky[i+1])){
	  kxAux = kx[i]; kx[i] = kx[i+1]; kx[i+1] = kxAux;
	  kxAux = ky[i]; ky[i] = ky[i+1]; ky[i+1] = kxAux;
	  kxAux = s[i];  s[i] = s[i+1];   s[i+1] = kxAux;
	  kxAux = k[i];  k[i] = k[i+1];   k[i+1] = kxAux;
	  m = i + 1;
	}
      }
      l = m;
    }

    

    string nameOutput;
    stringstream ss;
    ss << slice;
    nameOutput = name + "_" + ss.str() + ".dat";
    ofstream fileoutput(nameOutput.c_str());
    
    for(int i=0;i<elements;i++){
      //fileoutput << sqrt(ky[i]*ky[i]+kx[j]*kx[j]) << " " << s[i*n+j] << endl;
      fileoutput << (2./dx) * sqrt( sin(kx[i]*dx*0.5)*sin(kx[i]*dx*0.5) + sin(ky[i]*dx*0.5)*sin(ky[i]*dx*0.5) ) << "    ";
      fileoutput << s[i] << endl;
    }
    fileoutput.close();
  }

  
  fileinput.close();
  

}
