/** File:		sph_data.h
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

#ifndef __SPHDATA_H__
#define __SPHDATA_H__

#include "sph_header.h"

float window_width=600;
float window_height=600;

float xRot = 0.0f;
float yRot = 0.0f;
float xTrans = 0;
float yTrans = 0;
float zTrans = -36.0;

int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;

float3 real_world_origin;
float3 real_world_side;
float3 sim_ratio;

float world_width;
float world_height;
float world_length;

#endif
