// Filename: initializeRandomNumber.cu
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

bool initializeRandomNumber(){
  if(bool_seed==0){
    seed=time(0);
    srand(seed);
    seed = seed%18671;
  }

  cout << endl;
  RANTEST(seed);
  cout << endl;

  cout << "INITIALIZE RANDOM NUMBERS :     DONE" << endl;
  return 1;
}
