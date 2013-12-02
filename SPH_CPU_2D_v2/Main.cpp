/** File:		Main.cpp
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

#include <GL\glut.h>
#include <stdlib.h>
#include <stdio.h>
#include "SPHSystem.h"
#include "Timer.h"

int winX = 600;
int winY = 600;

Timer *sph_timer;
char *window_title;
SPHSystem *sph;

void init()
{
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluOrtho2D(0.0, sph->getWorldSize().x, 0.0, sph->getWorldSize().y);
	glViewport(0, 0, winX, winY);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	
	glClearColor(1.0f, 1.0f, 1.0f, 1.0);
}

void drawParticles()
{
	Particle *p = sph->getParticles();
	glColor3f(0.2f, 0.2f, 1.0f);
	glPointSize(5.0f);

	glBegin(GL_POINTS);
		for(uint i=0; i<sph->getNumParticle(); i++)
		{
			glVertex2f(p[i].pos.x, p[i].pos.y);
		}
	glEnd();
}

void displayFunc()
{
	sph->animation();

	glClear(GL_COLOR_BUFFER_BIT);
	drawParticles();
	glutSwapBuffers();

	sph_timer->update();
	memset(window_title, 0, 50);
	sprintf(window_title, "SPH System 2D. FPS: %0.2f", sph_timer->get_fps());
	glutSetWindowTitle(window_title);
}

void idleFunc()
{
	glutPostRedisplay();
}

void reshapeFunc(int width, int height)
{
	winX = width;
	winY = height;
	glViewport(0, 0, winX, winY);
	glutReshapeWindow(winX, winY);
}

int main(int argc, char **argv)
{
	sph = new SPHSystem();
	sph->initFluid();

	sph_timer=new Timer();
	window_title=(char *)malloc(sizeof(char)*50);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(winX, winY);
	glutCreateWindow("SPH Fluid 2D");
	init();

	glutDisplayFunc(displayFunc);
	glutReshapeFunc(reshapeFunc);
	glutIdleFunc(idleFunc);
    glutMainLoop();

	free(sph);
	return 0;
}
