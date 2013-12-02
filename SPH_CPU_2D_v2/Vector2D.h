/** File:		Vector2D.h
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

#ifndef __VECTOR2D_H__
#define __VECTOR2D_H__

#include <math.h>

typedef unsigned int uint;

class Vec2f
{
public:
	float x;
	float y;

public:
	Vec2f() {}
	Vec2f(float _x, float _y) { x=_x; y=_y; };
	Vec2f(Vec2f &vec) { x=vec.x; y=vec.y; };

	Vec2f operator + (const Vec2f &vec) const { return Vec2f(x+vec.x, y+vec.y); }
	Vec2f operator - (const Vec2f &vec) const { return Vec2f(x-vec.x, y-vec.y); }
	Vec2f operator * (const Vec2f &vec) const { return Vec2f(x*vec.x, y*vec.y); }
	Vec2f operator / (const Vec2f &vec) const { return Vec2f(x/vec.x, y/vec.y); }

	friend  Vec2f operator + (const Vec2f &vec, float n) { return Vec2f(vec.x+n, vec.y+n); }
	friend  Vec2f operator - (const Vec2f &vec, float n) { return Vec2f(vec.x-n, vec.y-n); }
	friend  Vec2f operator * (const Vec2f &vec, float n) { return Vec2f(vec.x*n, vec.y*n); }
	friend  Vec2f operator / (const Vec2f &vec, float n) { return Vec2f(vec.x/n, vec.y/n); }

	friend  Vec2f operator + (float n, const Vec2f &vec) { return Vec2f(n+vec.x, n+vec.y); }
	friend  Vec2f operator - (float n, const Vec2f &vec) { return Vec2f(n-vec.x, n-vec.y); }
	friend  Vec2f operator * (float n, const Vec2f &vec) { return Vec2f(n*vec.x, n*vec.y); }
	friend  Vec2f operator / (float n, const Vec2f &vec) { return Vec2f(n/vec.x, n/vec.y); }

	float Dot(const Vec2f &vec) const { return x*vec.x+y*vec.y; }
	Vec2f genNormal() const { return  *this / Length();} 
	void Normalize() { *this = *this / Length(); }

	float LengthSquared() const { return x*x+y*y; }
	float Length() const { return sqrt(LengthSquared()); }
};

class Vec2i
{
public:
	int x;
	int y;

public:
	Vec2i(){}
	Vec2i(int _x, int _y){ x=_x; y=_y; }
	Vec2i(Vec2i &vec) { x=vec.x; y=vec.y; }
};

class Vec2u
{
public:
	uint x;
	uint y;

public:
	Vec2u(){}
	Vec2u(uint _x, uint _y){ x=_x; y=_y; }
	Vec2u(Vec2u &vec) { x=vec.x; y=vec.y; }
};

#endif
