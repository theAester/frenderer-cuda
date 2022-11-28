#ifndef _FREDNDERER_OBJECT_H_
#define _FREDNDERER_OBJECT_H_
class object{
	public:
	vec3<double> pos;
	__host__ __device__ object(vec3<double>);
	object();
};
#endif
