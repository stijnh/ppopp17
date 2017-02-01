.PHONY: all clean

FLAGS=-O3 -Wall -Wextra -march=native -m64 -fopenmp
CPPFLAGS=$(FLAGS) -stdc++

NVFLAGS=-O3 -m64 std=c++11 -Xcompiler "$(CPPFLAGS)" \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_52,code=sm_52

BINS=hygraph

all: $(BINS)

hygraph: hygraph.cu
	nvcc $< -o $@ $(NVFLAGS)

clean:
	rm -rf $(BINS)
