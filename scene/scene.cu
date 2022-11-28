#include <cuda/std/complex>
#include <cuda/std/limits>
#include <cmath>
#include <list>
#include <limits>
#include <stdio.h>
#define inf cuda::std::numeric_limits<double>::max()
using namespace std;
using cplx = cuda::std::complex<double>;
#include "../common/vec3.h"
#include "../common/dir3.h"
#include "../object/object.h"
#include "../object/camera.h"
#include "../object/body.h"
#include "../object/light.h"
#include "scene.h"

#ifndef M_PI
#define M_PI 3.14159265368979
#endif

scene::scene(int resolution_x, int resolution_y, camera cam, vec3<double> ambient){
	resolution.x = resolution_x;
	resolution.y = resolution_y;
	this->cam = cam;
	this->ambient = ambient;
	this->flatbodies = NULL;
	this->flatlights = NULL;
}

__global__ void render_kernel(scene* sc, void* buff, int pitch, bool grey,
		double stepx, double stepy, dir3<double> khat, dir3<double> cam_up, vec3<double> screen_corner){
	int i= blockDim.x * blockIdx.x + threadIdx.x;
	int j= blockDim.y * blockIdx.y + threadIdx.y;
	if(i > sc->resolution.x || j > sc->resolution.y) return;
	printf("(%d,%d) flatbcount: %d, resx: %d, resy: %d, fov: %lf, r: %lf \n",i,j,sc->flatbodylen,sc->resolution.x, sc->resolution.y, sc->cam.fov, sc->flatbodies[0].angle.x);
	uint32_t* pp;
	unsigned char* kk;
	if(!grey) pp = (uint32_t*)((uint8_t*) buff + j*pitch);
	else kk = (unsigned char*)buff + j*pitch;

	dir3<double> prdir (screen_corner + khat*i*stepx - cam_up*j*stepy);
	vec3<double> acq_color = sc->raytrace_master(prdir);
	acq_color = sc->clip(acq_color);
	if(!grey) *pp= 0xFF000000 | ((uint8_t) (acq_color.x*255))<<16 | 
		((uint8_t) (acq_color.y*255))<<8 | (uint8_t) (acq_color.z*255);
	else *kk = (unsigned char) (acq_color.x*255);
}

void scene::render(void* screen_in,int pitch, bool grey){
	// calculate screen projection points
	int resolutionx = resolution.x/threadw +1;
	int resolutiony = resolution.y/threadh +1;
	double aspect_ratio = ((double)resolutionx)/((double)resolutiony);
	dir3<double> cam_proj = cam.proj_dir();
	dir3<double> cam_up = cam.up_dir();

	dir3<double> khat = dir3<double>(
			cam_proj.y * cam_up.z - cam_proj.z * cam_up.y,
			cam_proj.z * cam_up.x - cam_proj.x * cam_up.z,
			cam_proj.x * cam_up.y - cam_proj.y * cam_up.x
			);//cross product

	double dist = cam.lense_dist;
	double l = dist*tan(cam.fov*M_PI/360);
	double ll = l / aspect_ratio;
	
	vec3<double> screen_corner = cam_up*ll - khat*l + cam_proj*dist;

	double stepx = 2*l/resolution.x;
	double stepy = 2*ll/resolution.y;
	

	// copy the scene to GPU address space
	this->copyflatbodies();
	this->copyflatlights();

	scene* d_cs;
	cudaMalloc(&d_cs, sizeof(scene));
	cudaMemcpy(d_cs, this, sizeof(scene), cudaMemcpyHostToDevice);

	dim3 threads(threadw, threadh);
	dim3 blocks(resolutionx, resolutiony);

	unsigned char* d_buff;
	if(grey) cudaMalloc(&d_buff, resolution.x*resolution.y*sizeof(unsigned char));
	else cudaMalloc(&d_buff, resolution.x*resolution.y*sizeof(uint32_t));

	render_kernel<<<blocks, threads>>>(d_cs, d_buff, pitch, grey, stepx, stepy, khat, cam_up, screen_corner);
	cudaDeviceSynchronize();
	

	if(grey) cudaMemcpy(screen_in, d_buff, resolution.x*resolution.y*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	else cudaMemcpy(screen_in, d_buff, resolution.x*resolution.y*sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaFree(d_cs);
	cudaFree(d_buff);
	return;
}

__host__
void scene::copyflatbodies(){
	this->flatbodylen = bodies.size();
	if(flatbodies)
		cudaFree(flatbodies);
	cudaMalloc(&flatbodies, bodies.size()*sizeof(body));
	list<body*>::iterator itt = bodies.begin();
	list<body*>::iterator eitt = bodies.end();
	int i=0;
	while(itt != eitt){
		cudaMemcpy(flatbodies+i, *itt, sizeof(body), cudaMemcpyHostToDevice);
		itt++;
		i++;
	}
}

__host__
void scene::copyflatlights(){
	this->flatlightlen = lights.size();
	if(flatlights)
		cudaFree(flatlights);
	cudaMalloc(&flatlights, lights.size()*sizeof(light));
	list<light*>::iterator itt = lights.begin();
	list<light*>::iterator eitt = lights.end();
	int i=0;
	while(itt != eitt){
		cudaMemcpy(flatlights+i, *itt, sizeof(light), cudaMemcpyHostToDevice);
		itt++;
		i++;
	}
}

__device__
vec3<double> scene::clip(vec3<double> v){
	if(v.x<0) v.x = 0;
	else if(v.x>1) v.x = 1;
	if(v.y<0) v.y = 0;
	else if(v.y>1) v.y = 1;
	if(v.z<0) v.z = 0;
	else if(v.z>1) v.z = 1;
	return v;
}

__device__
vec3<double> scene::raytrace_master(dir3<double> dir){
	return raytrace_single(cam.pos, dir, 0);
}

__device__
vec3<double> scene::raytrace_single(vec3<double> pos, dir3<double> dir, int bounce =0){
	int inds=0;
	double mindist = inf;
	for(int i=0; i<flatbodylen; i++){
		double dist = flatbodies[i].intersect(pos,dir);
		if(dist != inf){ // collision detected
			if(dist<mindist){
				mindist = dist;
				inds = i;
			}
		}
	}
	if(mindist == inf)
		return vec3<double>(0,0,0); // into oblivion ...

	return flatbodies[inds].shader(this, mindist, dir, bounce + 1);
}
