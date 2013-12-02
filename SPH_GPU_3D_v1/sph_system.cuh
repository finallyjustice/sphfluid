/** File:		sph_system.cuh
 ** Author:		Dongli Zhang
 ** Contact:	dongli.zhang0129@gmail.com
 **
 ** Copyright (C) Dongli Zhang 2013
 **
 ** This program is free software;  you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation; either version 2 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY;  without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
 ** the GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program;  if not, write to the Free Software 
 ** Foundation, 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef __SPHSYSTEM_CU__
#define __SPHSYSTEM_CU__

#include "sph_header.h"
#include "sph_system.h"

#define CUDA_HOST_TO_DEV	1
#define CUDA_DEV_TO_HOST	2
#define CUDA_DEV_TO_DEV		3

void set_parameters(SysParam *hParam);
void alloc_array(void **dev_ptr, size_t size);
void free_array(void *dev_ptr);
void copy_array(void *ptr_a, void *ptr_b, size_t size, int type);
void copy_buffer(Particle *dMem, float4 *buffer, uint num_particle);
void compute_grid_size(int num_particle, int block_size, int &num_blocks, int &num_threads);
void calc_hash(uint *dHash, uint *dIndex, Particle *dMem, uint num_particle);
void sort_particles(uint *dHash, uint *dIndex, uint num_particle);
void find_start_end(uint *dStart, uint *dEnd, uint *dHash, uint *dIndex, uint num_particle, uint num_cell);
void compute(Particle *dMem, uint *dHash, uint *dIndex, uint *dStart, uint *dEnd, uint num_particle, uint tot_cell);
void integrate_velocity(Particle *dMem, uint num_particle);

#endif
