// Filename: s3d.cpp
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
//
// structureFactor inputFile nx ny dx
//
// with
// inputFile: name of the file to read for example
//
// nx / ny : number of cells in the x / y direction
//
// dx: grid spacing dx=h/ny

// s3d reads files with the structure factor
// generate by HydroGrid and writes the radial
// structure factor in the standard output
// k_discrete   S(k_discrete)
//
// with k_discrete = (2/dx) * sqrt( (sin(kx*dx/2))^2 + (sin(ky*dx/2))^2 )
//
//

#include <iostream>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>
using namespace std;


int main(  int argc, char* argv[]){

  
  string word;
  ifstream fileinput(argv[1]);
  int n = atoi(argv[2]);     //number of modes
  int ny = atoi(argv[3]);     //number of modes
  double dx = atof(argv[4]); // dx = lx / mx
  int elements = n*n;

  double kx[elements], ky[elements], k[elements], s[elements];
  double kxAux, kyAux, kAux, sAux;

  int min, max, middle, val;
  int m = 0;
  
  double kmin = 8 * atan(1.0) / (n*dx);

  getline(fileinput,word);
  getline(fileinput,word);

  for(int i=0;i<n;i++){
    //kx[i] = kmin * (i-n/2+1);
    fileinput >> kxAux;
    //cout << kx[i] << " " << ky[i] << endl;
    for(int j=0;j<n;j++){
      val = i*n + j;
      fileinput >> s[val];
      kx[val] = kmin * (i-n/2+1);
      ky[val] = kmin * (j-n/2+1);
      //k[val] = sqrt(kx[val]*kx[val] + ky[val]*ky[val]);
      k[val] = (2./dx) * sqrt( sin(kx[val]*dx*0.5)*sin(kx[val]*dx*0.5) + 
			       sin(ky[val]*dx*0.5)*sin(ky[val]*dx*0.5) );
    }
  }
  fileinput.close();

  // This loop sorts the wavenumbers: 
  n = elements;
  while(n>1){
    m=0;
    for(int i=0;i<n-1;i++){
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
    n = m;
  }

  cout.width(25);
  double kdiscrete;
  for(int i=0;i<elements;i++){
    //cout << k[i]  << "    " ;
    //cout << s[i]  << "    " ;
    //cout << kx[i] << "    " ;
    //cout << ky[i] << endl;

    kdiscrete = (2./dx) * sqrt( sin(kx[i]*dx*0.5)*sin(kx[i]*dx*0.5) +	\
                                sin(ky[i]*dx*0.5)*sin(ky[i]*dx*0.5) );
    if((kx[i]>=0) && (kdiscrete>1e-6))
    {
      cout << kdiscrete*(ny*dx) << " " << s[i] << endl;
    }

  }

}
