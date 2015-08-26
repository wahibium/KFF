// Filename: radialK.cpp
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
using namespace std;


int main(  int argc, char* argv[]){

  
  string word;
  ifstream fileinput(argv[1]);
  int n = atoi(argv[2]);     //number of modes
  double dx = atof(argv[3]); //lowest mode
  double dy = dx;
  int elements = n*n;
  
  getline(fileinput,word);
  getline(fileinput,word);
  
  
  double kx[n], ky[n];
  double s[n*n];
  
  for(int i=0;i<n;i++){
    kx[i] = dx * (i-n/2+1);
    fileinput >> ky[i];
    //cout << kx[i] << " " << ky[i] << endl;
    for(int j=0;j<n;j++)
      fileinput >> s[i*n+j];
  }
  
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++)
      cout << sqrt(kx[i]*kx[i]+kx[j]*kx[j]) << " " << s[i*n+j] << endl;
  }

  //for(int i=0;i<(n/2);i++){
  //for(int j=0;j<=i;j++){
  //cout << 0.1*sqrt(ky[n-i-1]*ky[n-i-1]+kx[n-j-1]*kx[n-j-1]) << " " << s[(n-i-1)*n+(n-j-1)] << endl;
  //cout << 0.1*sqrt(ky[i]*ky[i]+kx[j]*kx[j]) << " " << s[i*n+j] << endl;
  //    cout << i << " " << j << " " << ky[n-i-1] << " " << kx[n-j-1] << " " << 0.1*sqrt(ky[n-i-1]*ky[n-i-1]+kx[n-j-1]*kx[n-j-1]) << endl;
  //  cout << i << " " << j << " " << ky[i] << " " << kx[j] << " " << 0.1*sqrt(ky[i]*ky[i]+kx[j]*kx[j]) << endl;
  //}
  //}

  
  fileinput.close();
  

}
