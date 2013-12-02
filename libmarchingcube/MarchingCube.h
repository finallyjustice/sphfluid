/** File:		MarchingCube.h
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

#ifndef __MARCHINGCUBE_H__
#define __MARCHINGCUBE_H__

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned int uint;

struct float3
{
	float x;
	float y;
	float z;
};

class MarchingCube
{
private:

	uint row_vox;
	uint col_vox;
	uint len_vox;
	uint tot_vox;
	float step;

	float *scalar;
	float3 *normal;
	float3 *pos;
	float3 origin;

	float isovalue;

public:

	MarchingCube(uint _row_vox, uint _col_vox, uint _len_vox, float *_scalar, float3 *_pos, float3 _origin, float _step, float _isovalue);
	~MarchingCube();
	void run();
};

#endif
