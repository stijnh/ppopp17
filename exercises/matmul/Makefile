.PHONY: all clean

FLAGS=-O3 -g -Wall -Wextra -march=native -m64 -Wno-unused-parameter -Wno-unused-function -fopenmp
CFLAGS=$(FLAGS) -std=c11
CPPFLAGS=$(FLAGS) -std=c++11

NVFLAGS=-O3 -m64 -Xcompiler "$(CFLAGS)" \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_52,code=sm_52

STARPU_FLAGS=$$(pkg-config --cflags starpu-1.2)
STARPU_FLAGS+=$$(pkg-config --libs starpu-1.2)
STARPU_FLAGS+=-D_GNU_SOURCE

HPL_FLAGS=-I$(HPL_ROOT)/src -L$(HPL_ROOT)/lib -lHPL -lOpenCL

OBJ_FILES= framework.o \
		   matrix.o

BINS=sequential \
	 hpl \
	 cuda_openmp \
	 starpu \
	 ompss


all: $(BINS)

clean:
	rm -f $(BINS)

%.o: %.c
	gcc $< -c $(CFLAGS)

cuda_kernel.o: cuda_kernel.cu
	nvcc -c $< $(NVFLAGS)

sequential: sequential.c $(OBJ_FILES)
	gcc $^ -o $@ $(CFLAGS)

cuda_openmp: cuda_openmp.cu cuda_kernel.o $(OBJ_FILES)
	nvcc $^ -o $@ $(NVFLAGS)

ompss: ompss.c cuda_kernel.o $(OBJ_FILES)
	mcc $^ -o $@ $(CFLAGS) --ompss -lcudart

starpu: starpu.c cuda_kernel.o $(OBJ_FILES)
	gcc $^ -o $@ $(CFLAGS) $(STARPU_FLAGS)

hpl: hpl.cpp $(OBJ_FILES)
	g++ $^ -o $@ $(CPPFLAGS) $(HPL_FLAGS)
