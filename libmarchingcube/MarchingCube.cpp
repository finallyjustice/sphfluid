/** File:		MarchingCube.cpp
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

#include "MarchingCube.h"
#include "MarchingCubeTable.h"
#include <gl/glut.h>
#include <stdio.h>

MarchingCube::MarchingCube(uint _row_vox, uint _col_vox, uint _len_vox, float *_scalar, float3 *_pos, float3 _origin, float _step, float _isovalue)
{
	row_vox = _row_vox;
	col_vox = _col_vox;
	len_vox = _len_vox;
	tot_vox	= row_vox*col_vox*len_vox;

	scalar  = _scalar;
	pos		= _pos;
	origin	= _origin;

	isovalue=_isovalue;
	
	normal  = (float3 *)malloc(sizeof(float3)*tot_vox);
	memset(normal, 0, sizeof(float3)*tot_vox);
}

MarchingCube::~MarchingCube()
{
	free(normal);
}

void MarchingCube::run()
{
	float cube_value[8];
	float3 cube_pos[8];
	float3 cube_norm[8];

	float3 edge_vertex[12];
	float3 edge_norm[12];

	uint global_index;
	uint index;
	uint prev;
	uint next;
	uint x;
	uint y;
	uint z;

	int flag_index;
	int edge_flags;

	for(uint count_x=0; count_x<row_vox; count_x++)
	{
		for(uint count_y=0; count_y<col_vox; count_y++)
		{
			for(uint count_z=0; count_z<len_vox; count_z++)
			{
				global_index=count_z*row_vox*col_vox + count_y*row_vox + count_x;

				if(count_x == 0)
				{
					prev=count_z*row_vox*col_vox + count_y*row_vox + count_x+1;
					normal[global_index].x=(scalar[prev]-0.0f)/step;
				}

				if(count_x == row_vox-1)
				{
					next=count_z*row_vox*col_vox+ count_y*row_vox + count_x-1;
					normal[global_index].x=(0.0f-scalar[next])/step;
				}

				if(count_x!=0 && count_x!=row_vox-1)
				{
					prev=count_z*row_vox*col_vox + count_y*row_vox + count_x+1;
					next=count_z*row_vox*col_vox + count_y*row_vox + count_x-1;
					normal[global_index].x=(scalar[prev]-scalar[next])/step;
				}

				if(count_y == 0)
				{
					prev=count_z*row_vox*col_vox + (count_y+1)*row_vox + count_x;
					normal[global_index].y=(scalar[prev]-0.0f)/step;
				}

				if(count_y == col_vox-1)
				{
					next=count_z*row_vox*col_vox + (count_y-1)*row_vox + count_x;
					normal[global_index].y=(0.0f-scalar[next])/step;
				}

				if(count_y!=0 && count_y!=col_vox-1)
				{
					prev=count_z*row_vox*col_vox + (count_y+1)*row_vox + count_x;
					next=count_z*row_vox*col_vox + (count_y-1)*row_vox + count_x;
					normal[global_index].y=(scalar[prev]-scalar[next])/step;
				}

				if(count_z == 0)
				{
					prev=(count_z+1)*row_vox*col_vox + count_y*row_vox + count_x;
					normal[global_index].z=(scalar[prev]-0.0f)/step;
				}

				if(count_z == len_vox-1)
				{
					next=(count_z-1)*row_vox*col_vox + count_y*row_vox + count_x;
					normal[global_index].z=(0.0f-scalar[next])/step;
				}

				if(count_z!=0 && count_z!=len_vox-1)
				{
					prev=(count_z+1)*row_vox*col_vox + count_y*row_vox + count_x;
					next=(count_z-1)*row_vox*col_vox + count_y*row_vox + count_x;
					normal[global_index].z=(scalar[prev]-scalar[next])/step;
				}

				float norm=-sqrt(normal[global_index].x*normal[global_index].x+normal[global_index].y*normal[global_index].y+normal[global_index].z*normal[global_index].z);

				if(norm == 0.0f)
				{
					normal[global_index].x=0.0f;
					normal[global_index].y=0.0f;
					normal[global_index].z=0.0f;
				}
				else
				{
					normal[global_index].x=normal[global_index].x/norm;
					normal[global_index].y=normal[global_index].y/norm;
					normal[global_index].z=normal[global_index].z/norm;
				}
			}
		}
	}

	for(uint count_x=0; count_x<row_vox-1; count_x++)
	{
		for(uint count_y=0; count_y<col_vox-1; count_y++)
		{
			for(uint count_z=0; count_z<len_vox-1; count_z++)
			{
				global_index=count_z*row_vox*col_vox + count_y*row_vox + count_x;

				for(uint count=0; count<8; count++)
				{
					x=count_x+vertex_offset[count][0];
					y=count_y+vertex_offset[count][1];
					z=count_z+vertex_offset[count][2];

					index = z*row_vox*col_vox + y*row_vox + x;
					cube_value[count]=scalar[index];
					cube_pos[count]=pos[index];
					cube_norm[count]=normal[index];
				}

				flag_index=0;
				for(uint count=0; count<8; count++)
				{
					if(cube_value[count] < isovalue) 
					{
						flag_index |= 1<<count;
					}
				}

				edge_flags=cube_edge_flags[flag_index];
				if(edge_flags == 0) 
				{
					continue;
				}

				for(uint count=0; count<12; count++)
				{
					if(edge_flags & (1<<count))
					{
						float diff = (isovalue - cube_value[edge_conn[count][0]]) / (cube_value[edge_conn[count][1]] - cube_value[edge_conn[count][0]]);

						edge_vertex[count].x = cube_pos[edge_conn[count][0]].x + (cube_pos[edge_conn[count][1]].x - cube_pos[edge_conn[count][0]].x) * diff;
						edge_vertex[count].y = cube_pos[edge_conn[count][0]].y + (cube_pos[edge_conn[count][1]].y - cube_pos[edge_conn[count][0]].y) * diff;
						edge_vertex[count].z = cube_pos[edge_conn[count][0]].z + (cube_pos[edge_conn[count][1]].z - cube_pos[edge_conn[count][0]].z) * diff;

						edge_norm[count].x = cube_norm[edge_conn[count][0]].x + (cube_norm[edge_conn[count][1]].x - cube_norm[edge_conn[count][0]].x) * diff;
						edge_norm[count].y = cube_norm[edge_conn[count][0]].y + (cube_norm[edge_conn[count][1]].y - cube_norm[edge_conn[count][0]].y) * diff;
						edge_norm[count].z = cube_norm[edge_conn[count][0]].z + (cube_norm[edge_conn[count][1]].z - cube_norm[edge_conn[count][0]].z) * diff;
					}
				}

				for(uint count_triangle = 0; count_triangle < 5; count_triangle++)
				{
					if(triangle_table[flag_index][3*count_triangle] < 0)
					{
						break;
					}

					glBegin(GL_TRIANGLES);
						for(uint count_point = 0; count_point < 3; count_point++)
						{
							index = triangle_table[flag_index][3*count_triangle+count_point];

							glNormal3f(edge_norm[index].x, edge_norm[index].y, edge_norm[index].z);
							glVertex3f(edge_vertex[index].x+origin.x,
										edge_vertex[index].y+origin.y, 
										edge_vertex[index].z+origin.z);
						
						}
					glEnd();
				}
			}
		}
	}
}
