#include "../common/vec3.h"
#include "../common/dir3.h"
#include "object.h"

__host__ __device__
object::object(vec3<double> p){
	pos = p;
}
object::object(){}
