// Filename: hydroAnalysis.h
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


#ifdef GLOBALS_HYDROANALYSIS
#define EXTERN_HYDROANALYSIS 
#else
#define EXTERN_HYDROANALYSIS extern
#endif

EXTERN_HYDROANALYSIS int nCells[3];
EXTERN_HYDROANALYSIS double systemLength[3];
EXTERN_HYDROANALYSIS double heatCapacity[1];
EXTERN_HYDROANALYSIS double standardDesviation; // Standard deviation
EXTERN_HYDROANALYSIS double *velocities;
EXTERN_HYDROANALYSIS double *densities;
EXTERN_HYDROANALYSIS double *concent;

