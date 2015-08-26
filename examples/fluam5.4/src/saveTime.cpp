// Filename: saveTime.cpp
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


#include "header.h"
time_t t0, t1;

bool saveTime(int index){
  if(index==0){
    t0 = time(NULL);  
  }
  else if(index==2){
    t1 = time(NULL);
    t1 = t1-t0;
    cout << endl << "TIME RUN_GPU                       " << t1 << endl << endl;

    string NombreTime;
    NombreTime =  outputname + ".time";
    ofstream fileSaveTime(NombreTime.c_str());
    fileSaveTime << "TIME RUN GPU     " << t1 << endl;
    fileSaveTime.close();
  }
  else{
    return 0;
  }


  return 1;
}
