#ifndef _FRENDERER_dir3_H_
#define _FRENDERER_dir3_H_

template <class T>
class dir3{
	public:
	T x;
	T y;
	T z;
	__host__ __device__ dir3(T,T,T);
	__host__ __device__ dir3(vec3<T>);
	__host__ __device__ dir3();
	__host__ __device__ dir3<T> operator+(dir3<T>);
	__host__ __device__ dir3<T> operator-(dir3<T>);
	__host__ __device__ vec3<T> operator*(T);
	__host__ __device__ T operator*(dir3<T>);
	__host__ __device__ void operator+=(dir3<T>);
	__host__ __device__ void operator-=(dir3<T>);
	__host__ __device__ void pointTowards(vec3<T>);
};

#endif
