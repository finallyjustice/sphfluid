/** File:		sph_system.h
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

#ifndef __SPHSYSTEM_H__
#define __SPHSYSTEM_H__

#include "sph_header.h"
#include <cutil_math.h>

class SysParam
{
public:
	uint max_particle;
	uint num_particle;

	float kernel;
	float mass;

	float3 world_size;
	float cell_size;
	uint3 grid_size;
	uint tot_cell;

	float3 gravity;
	float wall_damping;
	float rest_density;
	float gas_constant;
	float viscosity;
	float time_step;
	float surf_normal;
	float surf_coe;

	float poly6_value;
	float spiky_value;
	float visco_value;

	float grad_poly6;
	float lplc_poly6;

	float kernel_2;
	float self_dens;
	float self_lplc_color;

	float3 sim_ratio;
	float3 sim_origin;
};

class Particle
{
public:
	float3 pos;
	float3 vel;
	float3 acc;
	float3 ev;

	float dens;
	float pres;

	float surf_norm;
};

class SPHSystem
{
public:
	Particle *hMem;
	Particle *dMem;
	float4 *hPoints;
	float4 *dPoints;

	uint *dHash;	
	uint *dIndex;
	uint *dStart;
	uint *dEnd;

	SysParam *hParam;

	uint sys_running;

public:
	SPHSystem();
	~SPHSystem();
	void animation();
	void init_system();
	void add_particle(float3 pos, float3 vel);
	void set_sim(float3 ratio, float3 origin);
};

#endif
