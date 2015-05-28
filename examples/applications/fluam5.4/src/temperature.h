// Filename: temperature.h
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


#ifdef TEMPERATURE
#define EXTERN_TEMPERATURE
#else
#define EXTERN_TEMPERATURE extern
#endif

EXTERN_TEMPERATURE double sigma_t_fluid, t_mean_fluid;
EXTERN_TEMPERATURE double txm, tym, tzm;
EXTERN_TEMPERATURE double t_mean_particle;
EXTERN_TEMPERATURE double txmP, tymP, tzmP;
