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
  int particle = atoi(argv[2]);
  
  getline(fileinput,word);
  word = word.substr(18,10);
  int np;
  np = atoi(word.c_str());
  cout << "#NUMBER PARTICLES " << np << ", PARTICLE " << particle << endl;
  
  float t, x, y, z, u;
  cout.precision(12);
  while(!fileinput.eof()){
    fileinput >> t;
    for(int i=0;i<np;i++){
      if(i==particle) fileinput >> x >> y >> z;
      else fileinput >> u >> u >> u;
    }
    cout << t << " " << x << " " << y << " " << z << endl;

  }

  fileinput.close();
}
