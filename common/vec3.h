#ifndef _FRENDERER_VEC3_H_
#define _FRENDERER_VEC3_H_

template<class T>
class dir3;

template<class T>
class vec3{
	public:
	T x;
	T y;
	T z;
	__host__ __device__ vec3(T,T,T);
	__host__ __device__ vec3();
	__host__ __device__ vec3(dir3<T>);

	__host__ __device__ vec3<T> operator+(vec3<T>);
	__host__ __device__ vec3<T> operator-(vec3<T>);
	__host__ __device__ T operator*(vec3<T>);
	__host__ __device__ vec3<T> operator*(T);

	__host__ __device__ void operator+=(vec3<T>);
	__host__ __device__ void operator-=(vec3<T>);
	__host__ __device__ void operator*=(T);

};
#endif
