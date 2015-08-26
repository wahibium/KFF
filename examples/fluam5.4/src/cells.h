// Filename: cells.h
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


#ifdef GLOBALS_CELLS
#define EXTERN_CELLS 
#else
#define EXTERN_CELLS extern
#endif

EXTERN_CELLS int mx;
EXTERN_CELLS int my;
EXTERN_CELLS int mz;
EXTERN_CELLS int ncells;
EXTERN_CELLS int mxt;
EXTERN_CELLS int myt;
EXTERN_CELLS int mzt;
EXTERN_CELLS int ncellst;
EXTERN_CELLS double lx, ly, lz;
EXTERN_CELLS double velocityboundary;
EXTERN_CELLS double cVolume;
EXTERN_CELLS double *cDensity;
EXTERN_CELLS double *crx, *cry, *crz;
EXTERN_CELLS double *cvx, *cvy, *cvz;
EXTERN_CELLS double vx0, vy0, vz0;
//EXTERN_CELLS int** cellneighbour;

typedef struct {
  int numberneighbours;
  int* cellneighbour;
  double volume;
  double** e;
  double mass, density;
  double* r;
  double* v;
  double* f;
} cell;

//EXTERN_CELLS cell* c;

