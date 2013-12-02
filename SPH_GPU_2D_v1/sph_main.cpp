/** File:		sph_main.cpp
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

#include "sph_header.h"
#include "sph_system.h"
#include "sph_timer.h"
#include <GL\glut.h>

Timer *sph_timer;
char *window_title;

float window_width;
float window_height;

SPHSystem *sph;

bool init_cuda(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) 
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	for(i = 0; i < count; i++) 
	{
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
		{
			if(prop.major >= 1) 
			{
				break;
			}
		}
	}
	if(i == count) 
	{
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}

	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

void  init_sph_system()
{
	sph=new SPHSystem();
	sph->init_system();

	sph_timer=new Timer();
	window_title=(char *)malloc(sizeof(char)*50);

	window_width=600;
	window_height=window_width/sph->hParam->world_size.x*sph->hParam->world_size.y;
}

void init()
{
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluOrtho2D(0.0, sph->hParam->world_size.x, 0.0, sph->hParam->world_size.y);
}

void display_particle()
{
	glColor3f(0.2f, 0.2f, 1.0f);
	glPointSize(2.0f);

	//glEnableClientState(GL_VERTEX_ARRAY);
	//glVertexPointer(2, GL_FLOAT, 0, sph->hPoints);
	//glDrawArrays(GL_POINTS, 0, sph->hParam->num_particle);

	for(uint i=0; i<sph->hParam->num_particle; i++)
	{
		if(sph->hMem[i].surf_norm > sph->hParam->surf_normal)
		{
			glColor3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			glColor3f(0.2f, 0.2f, 1.0f);
		}
		glBegin(GL_POINTS);
			glVertex2f(sph->hPoints[i].x, sph->hPoints[i].y);
		glEnd();
	}
}


void display_func()
{
	glViewport(0, 0, window_width, window_height);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	
	glClearColor(0.7f, 0.7f, 0.7f, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0f, 0.0f, 1.0f);
	
	sph->animation();
	display_particle();

	glutSwapBuffers();

	sph_timer->update();
	memset(window_title, 0, 50);
	sprintf(window_title, "SPH System 2D. FPS: %0.2f", sph_timer->get_fps());
	glutSetWindowTitle(window_title);
}

void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(int width, int height)
{
	window_width=width;
	window_height=height;
	glutReshapeWindow(window_width, window_height);
}

void process_keyboard(unsigned char key, int x, int y)
{
	if(key == ' ')
	{
		sph->sys_running=1-sph->sys_running;
	}
}

int main(int argc, char **argv)
{
	init_cuda();
	cudaDeviceProp prop;
    int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice( &dev, &prop );

	glutInit(&argc, argv);

	init_sph_system();

	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("SPH System 2D");

	init();

    glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutIdleFunc(idle_func);
	glutKeyboardFunc(process_keyboard);
    glutMainLoop();

	free(sph);

	return 0;
}
