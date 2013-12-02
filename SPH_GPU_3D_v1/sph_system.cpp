/** File:		sph_system.cpp
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

#include "sph_system.h"
#include "sph_system.cuh"

SPHSystem::SPHSystem()
{
	hParam=new SysParam();

	hParam->max_particle=25000;
	hParam->num_particle=0;

	hParam->kernel=0.04f;
	hParam->mass=0.02f;

	hParam->world_size=make_float3(0.64f, 0.64f, 0.64f);
	hParam->cell_size=hParam->kernel;
	hParam->grid_size.x=(uint)ceil(hParam->world_size.x/hParam->cell_size);
	hParam->grid_size.y=(uint)ceil(hParam->world_size.y/hParam->cell_size);
	hParam->grid_size.z=(uint)ceil(hParam->world_size.z/hParam->cell_size);
	hParam->tot_cell=hParam->grid_size.x*hParam->grid_size.y*hParam->grid_size.z;

	hParam->gravity.x=0.0f; 
	hParam->gravity.y=-8.0f;
	hParam->gravity.z=0.0f; 
	hParam->wall_damping=-0.5f;
	hParam->rest_density=1000.0f;
	hParam->gas_constant=1.0f;
	hParam->viscosity=6.5f;
	hParam->time_step=0.003f;
	hParam->surf_normal=3.0f;
	hParam->surf_coe=0.2f;

	hParam->poly6_value=315.0f/(64.0f * PI * pow(hParam->kernel, 9));;
	hParam->spiky_value=-45.0f/(PI * pow(hParam->kernel, 6));
	hParam->visco_value=45.0f/(PI * pow(hParam->kernel, 6));

	hParam->grad_poly6=-945/(32 * PI * pow(hParam->kernel, 9));
	hParam->lplc_poly6=-945/(8 * PI * pow(hParam->kernel, 9));

	hParam->kernel_2=hParam->kernel*hParam->kernel;
	hParam->self_dens=hParam->mass*hParam->poly6_value*pow(hParam->kernel, 6);
	hParam->self_lplc_color=hParam->lplc_poly6*hParam->mass*(hParam->kernel_2)*(0-3/4*(hParam->kernel_2));

	sys_running=0;

	hMem=(Particle *)malloc(sizeof(Particle)*hParam->max_particle);
	alloc_array((void**)&(dMem), sizeof(Particle)*hParam->max_particle);
	hPoints=(float4 *)malloc(sizeof(float4)*hParam->max_particle);
	alloc_array((void**)&(dPoints), sizeof(float4)*hParam->max_particle);

	alloc_array((void**)&dHash, sizeof(uint)*hParam->max_particle);
	alloc_array((void**)&dIndex, sizeof(uint)*hParam->max_particle);
	alloc_array((void**)&dStart, sizeof(uint)*hParam->tot_cell);
	alloc_array((void**)&dEnd, sizeof(uint)*hParam->tot_cell);

	printf("Initialize SPH:\n");
	printf("World Width : %f\n", hParam->world_size.x);
	printf("World Height: %f\n", hParam->world_size.y);
	printf("World Length: %f\n", hParam->world_size.z);
	printf("Cell Size  : %f\n", hParam->cell_size);
	printf("Grid Width : %u\n", hParam->grid_size.x);
	printf("Grid Height: %u\n", hParam->grid_size.y);
	printf("Grid Length: %u\n", hParam->grid_size.y);
	printf("Poly6 Kernel: %f\n", hParam->poly6_value);
	printf("Spiky Kernel: %f\n", hParam->spiky_value);
	printf("Visco Kernel: %f\n", hParam->visco_value);
	printf("Self Density: %f\n", hParam->self_dens);
}

SPHSystem::~SPHSystem()
{
	free(hMem);
	free(hParam);
	free(hPoints);

	free_array(dMem);
	free_array(dHash);
	free_array(dIndex);
	free_array(dStart);
	free_array(dEnd);
	free_array(dPoints);
}

void SPHSystem::animation()
{	
	if(sys_running != 1)
	{
		return;
	}

	set_parameters(hParam);

    calc_hash(dHash, dIndex,dMem, hParam->num_particle);
	sort_particles(dHash, dIndex, hParam->num_particle);
	find_start_end(dStart, dEnd, dHash, dIndex, hParam->num_particle, hParam->tot_cell);
	compute(dMem, dHash, dIndex, dStart, dEnd, hParam->num_particle, hParam->tot_cell);
	integrate_velocity(dMem, hParam->num_particle);

	copy_buffer(dMem, dPoints, hParam->num_particle);
	copy_array(hPoints, dPoints, sizeof(float4)*hParam->num_particle, CUDA_DEV_TO_HOST);
	copy_array(hMem, dMem, sizeof(Particle)*hParam->num_particle, CUDA_DEV_TO_HOST);
}

void SPHSystem::add_particle(float3 pos, float3 vel)
{
	Particle *p=&(hMem[hParam->num_particle]);

	p->pos=pos;
	p->vel=vel;
	p->acc=make_float3(0.0f, 0.0f, 0.0f);
	p->ev=make_float3(0.0f, 0.0f, 0.0f);

	p->dens=hParam->rest_density;
	p->pres=0.0f;

	hParam->num_particle++;
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void SPHSystem::init_system()
{
	srand((unsigned int)time(NULL));

	float3 pos;
	float3 vel=make_float3(0.0f, 0.0f, 0.0f);

	for(pos.x=hParam->world_size.x*0.0f; pos.x<hParam->world_size.x*0.6f; pos.x+=(hParam->kernel*0.5f))
	{
		for(pos.y=hParam->world_size.y*0.0f; pos.y<hParam->world_size.y*0.8f; pos.y+=(hParam->kernel*0.5f))
		{
			for(pos.z=hParam->world_size.z*0.0f; pos.z<hParam->world_size.z*0.6f; pos.z+=(hParam->kernel*0.5f))
			{
				add_particle(pos, vel);
			}
		}
	}

	copy_array(dMem, hMem, sizeof(Particle)*hParam->num_particle, CUDA_HOST_TO_DEV);
	printf("Init Particle: %u\n", hParam->num_particle);
}

void SPHSystem::set_sim(float3 ratio, float3 origin)
{
	hParam->sim_ratio=ratio;
	hParam->sim_origin=origin;

}
