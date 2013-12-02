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
#include "sph_header.h"

SPHSystem::SPHSystem()
{
	max_particle=10000;
	num_particle=0;

	kernel=0.04f;
	mass=0.02f;

	world_size.x=1.28f;
	world_size.y=1.28f;
	cell_size=kernel;
	grid_size.x=(uint)ceil(world_size.x/cell_size);
	grid_size.y=(uint)ceil(world_size.y/cell_size);
	tot_cell=grid_size.x*grid_size.y;

	gravity.x=0.0f; 
	gravity.y=-1.8f;
	wall_damping=-0.5f;
	rest_density=1000.0f;
	gas_constant=1.0f;
	viscosity=6.5f;
	time_step=0.003f;
	surf_norm=6.0f;
	surf_coe=0.2f;

	poly6_value=315.0f/(64.0f * PI * pow(kernel, 9));;
	spiky_value=-45.0f/(PI * pow(kernel, 6));
	visco_value=45.0f/(PI * pow(kernel, 6));

	grad_poly6=-945/(32 * PI * pow(kernel, 9));
	lplc_poly6=-945/(8 * PI * pow(kernel, 9));

	kernel_2=kernel*kernel;
	self_dens=mass*poly6_value*pow(kernel, 6);
	self_lplc_color=lplc_poly6*mass*kernel_2*(0-3/4*kernel_2);

	mem=(Particle *)malloc(sizeof(Particle)*max_particle);
	cell=(Particle **)malloc(sizeof(Particle *)*tot_cell);

	sys_running=0;

	printf("Initialize SPH:\n");
	printf("World Width : %f\n", world_size.x);
	printf("World Height: %f\n", world_size.y);
	printf("Cell Size  : %f\n", cell_size);
	printf("Grid Width : %u\n", grid_size.x);
	printf("Grid Height: %u\n", grid_size.y);
	printf("Total Cell : %u\n", tot_cell);
	printf("Poly6 Kernel: %f\n", poly6_value);
	printf("Spiky Kernel: %f\n", spiky_value);
	printf("Visco Kernel: %f\n", visco_value);
	printf("Self Density: %f\n", self_dens);
}

SPHSystem::~SPHSystem()
{
	free(mem);
	free(cell);
}

void SPHSystem::animation()
{
	if(sys_running == 0)
	{
		return;
	}

	build_table();
	comp_dens_pres();
	comp_force_adv();
	advection();
}

void SPHSystem::init_system()
{
	float2 pos;
	float2 vel;

	vel.x=0.0f;
	vel.y=0.0f;

	for(pos.x=world_size.x*0.0f+kernel/2; pos.x<world_size.x*0.6f-kernel/2; pos.x+=(kernel*0.45f))
	{
		for(pos.y=world_size.y*0.0f+kernel/2; pos.y<world_size.y*0.6f-kernel/2; pos.y+=(kernel*0.45f))
		{
			add_particle(pos, vel);
		}
	}

	printf("Init Particle: %u\n", num_particle);
}

void SPHSystem::add_particle(float2 pos, float2 vel)
{
	Particle *p=&(mem[num_particle]);

	p->id=num_particle;

	p->pos=pos;
	p->vel=vel;

	p->acc.x=0.0f;
	p->acc.y=0.0f;
	p->ev.x=0.0f;
	p->ev.y=0.0f;

	p->dens=rest_density;
	p->pres=0.0f;

	p->next=NULL;

	num_particle++;
}

void SPHSystem::build_table()
{
	Particle *p;
	uint hash;

	for(uint i=0; i<tot_cell; i++)
	{
		cell[i]=NULL;
	}

	for(uint i=0; i<num_particle; i++)
	{
		p=&(mem[i]);
		hash=calc_cell_hash(calc_cell_pos(p->pos));

		if(cell[hash] == NULL)
		{
			p->next=NULL;
			cell[hash]=p;
		}
		else
		{
			p->next=cell[hash];
			cell[hash]=p;
		}
	}
}

void SPHSystem::comp_dens_pres()
{
	Particle *p;
	Particle *np;

	int2 cell_pos;
	int2 near_pos;
	uint hash;

	float2 rel_pos;
	float r2;

	for(uint i=0; i<num_particle; i++)
	{
		p=&(mem[i]); 
		cell_pos=calc_cell_pos(p->pos);

		p->dens=0.0f;
		p->pres=0.0f;

		for(int x=-1; x<=1; x++)
		{
			for(int y=-1; y<=1; y++)
			{
				near_pos.x=cell_pos.x+x;
				near_pos.y=cell_pos.y+y;
				hash=calc_cell_hash(near_pos);

				if(hash == 0xffffffff)
				{
					continue;
				}

				np=cell[hash];
				while(np != NULL)
				{
					rel_pos.x=np->pos.x-p->pos.x;
					rel_pos.y=np->pos.y-p->pos.y;
					r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y;

					if(r2<INF || r2>=kernel_2)
					{
						np=np->next;
						continue;
					}

					p->dens=p->dens + mass * poly6_value * pow(kernel_2-r2, 3);

					np=np->next;
				}
			}
		}

		p->dens=p->dens+self_dens;
		p->pres=(pow(p->dens / rest_density, 7) - 1) *gas_constant;
	}
}

void SPHSystem::comp_force_adv()
{
	Particle *p;
	Particle *np;

	int2 cell_pos;
	int2 near_pos;
	uint hash;

	float2 rel_pos;
	float2 rel_vel;

	float r2;
	float r;
	float kernel_r;
	float V;

	float pres_kernel;
	float visc_kernel;
	float temp_force;

	float2 grad_color;
	float lplc_color;

	for(uint i=0; i<num_particle; i++)
	{
		p=&(mem[i]); 
		cell_pos=calc_cell_pos(p->pos);

		p->acc.x=0.0f;
		p->acc.y=0.0f;

		grad_color.x=0.0f;
		grad_color.y=0.0f;
		lplc_color=0.0f;
		
		for(int x=-1; x<=1; x++)
		{
			for(int y=-1; y<=1; y++)
			{
				near_pos.x=cell_pos.x+x;
				near_pos.y=cell_pos.y+y;
				hash=calc_cell_hash(near_pos);

				if(hash == 0xffffffff)
				{
					continue;
				}

				np=cell[hash];
				while(np != NULL)
				{
					rel_pos.x=p->pos.x-np->pos.x;
					rel_pos.y=p->pos.y-np->pos.y;
					r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y;

					if(r2 < kernel_2 && r2 > INF)
					{
						r=sqrt(r2);
						V=mass/np->dens/2;
						kernel_r=kernel-r;

						pres_kernel=spiky_value * kernel_r * kernel_r;
						temp_force=V * (p->pres+np->pres) * pres_kernel;
						p->acc.x=p->acc.x-rel_pos.x*temp_force/r;
						p->acc.y=p->acc.y-rel_pos.y*temp_force/r;

						rel_vel.x=np->ev.x-p->ev.x;
						rel_vel.y=np->ev.y-p->ev.y;

						visc_kernel=visco_value*(kernel-r);
						temp_force=V * viscosity * visc_kernel;
						p->acc.x=p->acc.x + rel_vel.x*temp_force; 
						p->acc.y=p->acc.y + rel_vel.y*temp_force; 

						float temp=(-1) * grad_poly6 * V * pow(kernel_2-r2, 2);
						grad_color.x += temp * rel_pos.x;
						grad_color.y += temp * rel_pos.y;
						lplc_color += lplc_poly6 * V * (kernel_2-r2) * (r2-3/4*(kernel_2-r2));
					}

					np=np->next;
				}
			}
		}

		lplc_color+=self_lplc_color/p->dens;
		p->surf_norm=sqrt(grad_color.x*grad_color.x+grad_color.y*grad_color.y);

		if(p->surf_norm > surf_norm)
		{
			p->acc.x+=surf_coe * lplc_color * grad_color.x / p->surf_norm;
			p->acc.y+=surf_coe * lplc_color * grad_color.y / p->surf_norm;
		}
	}
}

void SPHSystem::advection()
{
	Particle *p;
	for(uint i=0; i<num_particle; i++)
	{
		p=&(mem[i]);

		p->vel.x=p->vel.x+p->acc.x*time_step/p->dens+gravity.x*time_step;
		p->vel.y=p->vel.y+p->acc.y*time_step/p->dens+gravity.y*time_step;

		p->pos.x=p->pos.x+p->vel.x*time_step;
		p->pos.y=p->pos.y+p->vel.y*time_step;

		if(p->pos.x >= world_size.x)
		{
			p->vel.x=p->vel.x*wall_damping;
			p->pos.x=world_size.x-BOUNDARY;
		}

		if(p->pos.x < 0.0f)
		{
			p->vel.x=p->vel.x*wall_damping;
			p->pos.x=0.0f;
		}

		if(p->pos.y >= world_size.y)
		{
			p->vel.y=p->vel.y*wall_damping;
			p->pos.y=world_size.y-BOUNDARY;
		}

		if(p->pos.y < 0.0f)
		{
			p->vel.y=p->vel.y*wall_damping;
			p->pos.y=0.0f;
		}

		p->ev.x=(p->ev.x+p->vel.x)/2;
		p->ev.y=(p->ev.y+p->vel.y)/2;
	}
}

int2 SPHSystem::calc_cell_pos(float2 p)
{
	int2 cell_pos;
	cell_pos.x = int(floor((p.x) / cell_size));
	cell_pos.y = int(floor((p.y) / cell_size));

    return cell_pos;
}

uint SPHSystem::calc_cell_hash(int2 cell_pos)
{
	if(cell_pos.x<0 || cell_pos.x>=(int)grid_size.x || cell_pos.y<0 || cell_pos.y>=(int)grid_size.y)
	{
		return (uint)0xffffffff;
	}

	cell_pos.x = cell_pos.x & (grid_size.x-1);  
    cell_pos.y = cell_pos.y & (grid_size.y-1);  

	return ((uint)(cell_pos.y))*grid_size.x+(uint)(cell_pos.x);
}
