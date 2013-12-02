/** File:		main.cpp
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
#include <gl\glut.h>

float window_width  = 600;
float window_height = 600;

float x_rot = 0.0f;
float y_rot = 0.0f;
float x_trans = 0;
float y_trans = 0;
float z_trans = -36.0;

int ox;
int oy;
int buttonState;
float x_rot_len = 0.0f;
float y_rot_len = 0.0f;

float3 world_origin;
float3 world_side;

float light_ambient[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
float light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
float light_position[] = { -10.0f, -10.0f, -10.0f, 1.0f };

float3 *model_vox;
float *model_scalar;
uint row_vox;
uint col_vox;
uint len_vox;
uint tot_vox;
float vox_size;
MarchingCube *mc;

void init_marching_cube_model()
{
	vox_size = 0.5f;
	row_vox = world_side.x/vox_size;
	col_vox = world_side.y/vox_size;
	len_vox = world_side.z/vox_size;
	tot_vox = row_vox*col_vox*len_vox;

	model_vox = (float3 *)malloc(sizeof(float3)*row_vox*col_vox*len_vox);
	model_scalar = (float *)malloc(sizeof(float)*row_vox*col_vox*len_vox);

	float model_radius = 8.0f;
	float3 model_center;

	model_center.x = 10.0f;
	model_center.y = 10.0f;
	model_center.z = 10.0f;

	uint index;
	float dist;
	for(uint count_x=0; count_x<row_vox; count_x++)
	{
		for(uint count_y=0; count_y<col_vox; count_y++)
		{
			for(uint count_z=0; count_z<len_vox; count_z++)
			{
				index = count_z*row_vox*col_vox+count_y*row_vox+count_x;

				model_vox[index].x = count_x*vox_size;
				model_vox[index].y = count_y*vox_size;
				model_vox[index].z = count_z*vox_size;

				dist = (model_vox[index].x-model_center.x)*(model_vox[index].x-model_center.x)
						+(model_vox[index].y-model_center.y)*(model_vox[index].y-model_center.y)
						+(model_vox[index].z-model_center.z)*(model_vox[index].z-model_center.z);

				if (dist > model_radius*model_radius)
				{
					model_scalar[index] = 0.0f;
					continue;
				}

				model_scalar[index] = pow(1.0-dist/model_radius/model_radius, 3);
			}
		}
	}

	mc = new MarchingCube(row_vox, col_vox, len_vox, model_scalar, model_vox, world_origin, vox_size, 0.4f);
}

void draw_box(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(1.0f);

    glBegin(GL_LINES);   
        
		//1
        glVertex3f(ox, oy, oz);
        glVertex3f(ox+width, oy, oz);

		//2
        glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy+height, oz);

		//3
        glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy, oz+length);

		//4
        glVertex3f(ox+width, oy, oz);
        glVertex3f(ox+width, oy+height, oz);

		//5
        glVertex3f(ox+width, oy+height, oz);
        glVertex3f(ox, oy+height, oz);

		//6
        glVertex3f(ox, oy+height, oz+length);
        glVertex3f(ox, oy, oz+length);

		//7
        glVertex3f(ox, oy+height, oz+length);
        glVertex3f(ox, oy+height, oz);

		//8
        glVertex3f(ox+width, oy, oz);
        glVertex3f(ox+width, oy, oz+length);

		//9
        glVertex3f(ox, oy, oz+length);
        glVertex3f(ox+width, oy, oz+length);

		//10
        glVertex3f(ox+width, oy+height, oz);
        glVertex3f(ox+width, oy+height, oz+length);

		//11
        glVertex3f(ox+width, oy+height, oz+length);
        glVertex3f(ox+width, oy, oz+length);

		//12
        glVertex3f(ox, oy+height, oz+length);
        glVertex3f(ox+width, oy+height, oz+length);
    glEnd();
}

void init()
{
	world_origin.x = -10.0f;
	world_origin.y = -10.0f;
	world_origin.z = -10.0f;

	world_side.x = 20.0f;
	world_side.y = 20.0f;
	world_side.z = 20.0f;

	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)window_width/window_height, 10.0f, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);	

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SMOOTH);
}

void display_func()
{
	glClearColor(0.75f, 0.75f, 0.75f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushMatrix();

	if (buttonState == 1)
	{
		x_rot += (x_rot_len-x_rot)*0.1f;
		y_rot += (y_rot_len-y_rot)*0.1f;
	}

	glTranslatef(x_trans, y_trans, z_trans);
    glRotatef(x_rot, 1.0f, 0.0f, 0.0f);
    glRotatef(y_rot, 0.0f, 1.0f, 0.0f);	

	glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION,light_position);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHTING);

	mc->run();

	glDisable(GL_LIGHTING);

	glColor3f(1.0f, 0.0f, 0.0f);
	draw_box(world_origin.x, world_origin.y, world_origin.z, world_side.x, world_side.y, world_side.z);

	glColor3f(1.0, 1.0, 0.0f);
	glPointSize(20.0f);
	glBegin(GL_POINTS);
		glVertex3f( 10.0f, 10.0f, 10.0f);
	glEnd();

	glPopMatrix();

    glutSwapBuffers();
}

void idle_func()
{
	//glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
	window_width  = width;
	window_height = height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)width/height, 0.001, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
	if(key == 'w')
	{
		z_trans += 1.0f;
	}

	if(key == 's')
	{
		z_trans -= 1.0f;
	}

	if(key == 'a')
	{
		x_trans -= 1.0f;
	}

	if(key == 'd')
	{
		x_trans += 1.0f;
	}

	if(key == 'q')
	{
		y_trans -= 1.0f;
	}

	if(key == 'e')
	{
		y_trans += 1.0f;
	}

	glutPostRedisplay();
}

void specialkey_func(int key, int x, int y)
{
	if(key == GLUT_KEY_UP)
	{
		z_trans += 1.0f;
	}

	if(key == GLUT_KEY_DOWN)
	{
		z_trans -= 1.0f;
	}

	if(key == GLUT_KEY_LEFT)
	{
		x_trans -= 1.0f;
	}

	if(key == GLUT_KEY_RIGHT)
	{
		x_trans += 1.0f;
	}

	glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
        buttonState = 1;
	}
    else if (state == GLUT_UP)
	{
        buttonState = 0;
	}

    ox = x; oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    float dx;
	float dy;

    dx = (float)(x - ox);
    dy = (float)(y - oy);

	if (buttonState == 1) 
	{
		x_rot_len += dy / 5.0f;
		y_rot_len += dx / 5.0f;
	}

	ox = x; 
	oy = y;

	glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Marching Cube");

	init();
	init_marching_cube_model();

    glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutSpecialFunc(specialkey_func);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);

    glutMainLoop();

	free(model_scalar);
	free(model_vox);

    return 0;
}
