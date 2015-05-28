// Filename: move.h
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


#ifdef GLOBALS_MOVE
#define EXTERN_MOVE 
#else
#define EXTERN_MOVE extern
#endif


EXTERN_MOVE double*** stress;
EXTERN_MOVE double*** dnoise_s;
EXTERN_MOVE double** dp;
EXTERN_MOVE double* dm;
EXTERN_MOVE double* dnoise_tr;



/*type(pointer_to_matrix), pointer :: stress(:), dnoise_s(:)
  type(pointer_to_vector), pointer :: dp(:)
  real, pointer :: dm(:), dnoise_tr(:)
  integer :: ruido*/


