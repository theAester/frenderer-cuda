#ifndef _FRENDERER_SCENE_H_
#define _FRENDERER_SCENE_H_
class scene{
	public:
	struct res{
		int x;
		int y;
	};
	__device__ vec3<double> raytrace_master(dir3<double>);
	__device__ vec3<double> clip(vec3<double>);
	static const int threadw = 16;
	static const int threadh = 10;
	void copyflatbodies();
	void copyflatlights();
	body* flatbodies;
	light* flatlights;
	int flatbodylen;
	int flatlightlen;
	void listbodies();
	void listlights();
	__device__ vec3<double> raytrace_single(vec3<double>, dir3<double>, int);
	scene(int, int, camera,vec3<double>);
	res resolution;
	__host__ void render(void*, int,bool=false);
	camera cam;
	list<body*> bodies;
	list<light*> lights;
	vec3<double> ambient;
};
#endif
