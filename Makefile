
CC=g++
CUDA_INC_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/include

.SECONDEXPANSION:
main%: build/vec3.o build/dir3.o build/camera.o build/object.o build/light.o build/body.o build/scene.o build/$$@.o
	$(CC) build/vec3.o build/dir3.o build/object.o build/camera.o build/light.o build/body.o build/scene.o build/$@.o -o build/$@

build/vec3.o:
	$(CC)  -g -dc common/vec3.cu -o build/vec3.o -I$(CUDA_INC_PATH)
build/dir3.o:
	$(CC)  -g -dc common/dir3.cu -o build/dir3.o -I$(CUDA_INC_PATH)
build/camera.o:
	$(CC)  -g -dc object/camera.cu -o build/camera.o -I$(CUDA_INC_PATH)
build/object.o:
	$(CC)  -g -dc object/object.cu -o build/object.o -I$(CUDA_INC_PATH)
build/light.o:
	$(CC)  -g -dc object/light.cu -o build/light.o -I$(CUDA_INC_PATH)
build/body.o:
	$(CC)  -g -dc object/body.cu -o build/body.o -I$(CUDA_INC_PATH)
build/scene.o:
	$(CC)  -g -dc scene/scene.cu -o build/scene.o -I$(CUDA_INC_PATH)

build/main%.o:
	$(eval part=`echo $@ | sed -E 's/.*\/(.*).o/\1/'`)
	$(CC)  -g $(part).cu -dc -o $@ -I$(CUDA_INC_PATH)

clean:
	rm build/*.o

