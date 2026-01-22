/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CONSTANTSH
#define CONSTANTSH

// Release number
#define RELEASE "2.2-H200"

// Use symmetry - reduces expected ops by sqrt(2) (~41% speedup)
#define USE_SYMMETRY

// Number of random jumps
// Max 512 for the GPU
// H200 optimization: Use 64 jumps for better cache utilization
#define NB_JUMP 64

// GPU group size
// H200 optimization: 256 for better occupancy on Hopper architecture
#define GPU_GRP_SIZE 256

// GPU number of run per kernel call
// H200 optimization: 256 runs for better throughput
#define NB_RUN 256

// Kangaroo type
#define TAME 0  // Tame kangaroo
#define WILD 1  // Wild kangaroo

// SendDP Period in sec
#define SEND_PERIOD 2.0

// Timeout before closing connection idle client in sec
#define CLIENT_TIMEOUT 3600.0

// Number of merge partition
#define MERGE_PART 256

#endif //CONSTANTSH
