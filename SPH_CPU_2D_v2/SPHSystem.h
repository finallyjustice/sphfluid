/** File:		SPHSystem.h
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

#include "Vector2D.h"
#include "Structure.h"

#define PI 3.141592f
#define INF 1E-12f

class SPHSystem
{
public:
	SPHSystem();
	~SPHSystem();
	void initFluid();
	void addSingleParticle(Vec2f pos, Vec2f vel);
	Vec2i calcCellPos(Vec2f pos);
	uint calcCellHash(Vec2i pos);

	//kernel function
	float poly6(float r2){ return 315.0f/(64.0f * PI * pow(kernel, 9)) * pow(kernel*kernel-r2, 3); }
	float spiky(float r){ return -45.0f/(PI * pow(kernel, 6)) * (kernel-r) * (kernel-r); }
	float visco(float r){ return 45.0f/(PI * pow(kernel, 6)) * (kernel-r); }

	//animation
	void compTimeStep();
	void buildGrid();
	void compDensPressure();
	void compForce();
	void advection();
	void animation();

	//getter
	uint getNumParticle(){ return numParticle; }
	Vec2f getWorldSize(){ return worldSize; }
	Particle* getParticles(){ return particles; }
	Cell* getCells(){ return cells; }


private:
	float kernel;
	float mass;

	uint maxParticle;
	uint numParticle;

	Vec2i gridSize;
	Vec2f worldSize;
	float cellSize;
	uint totCell;

	//params
	Vec2f gravity;
	float stiffness;
	float restDensity;
	float timeStep;
	float wallDamping;
	float viscosity;

	Particle *particles;
	Cell *cells;
};

#endif
