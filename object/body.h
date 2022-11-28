#ifndef _FRENDERER_BODY_H_
#define _FRENDERER_BODY_H_
class scene;
class body : public object{
	public:
	const int pdegree; 
	double reflectivity;
	__host__ __device__ body(vec3<double>, vec3<double>,double,int,bool=false,double=0,double=0,double=0); // done
	__host__ __device__ virtual vec3<double> diffuse_color(vec3<double>)=0;
	__host__ __device__ virtual cplx surf_func(vec3<cplx>)=0;
	__host__ __device__ virtual vec3<double> shader(scene*, double, dir3<double>, int)=0;
	__host__ __device__ vec3<double> lambert_shader(scene*, double, dir3<double>); // done
//	dir3<double> phong_shader(scene&, double, dir3<double>, int, double); // todo
	__host__ __device__ virtual double intersect(vec3<double>, dir3<double>); // done
	__host__ __device__ void change_angle(vec3<double>); // done
	__host__ __device__  virtual dir3<double> get_normal(vec3<double>);
	bool hitbox;
	double a;
	double b;
	double c;
	__host__ __device__  bool intersect_hitbox(vec3<double>,dir3<double>);
	__host__ __device__  vec3<cplx> TG(vec3<cplx>);
	__host__ __device__  vec3<double> TG(vec3<double>);
	__host__ __device__  cplx g(vec3<double>, dir3<double>, cplx); // done
	vec3<double> angle;
	double cosa;
	double sina;
	double cosb;
	double sinb;
	double cosc;
	double sinc;
	double R[3][3];
	private:
};

struct device_data{
	body* d_body;
	dir3<double>* d_dirs;
};
#endif
