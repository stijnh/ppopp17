#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "cuda_kernel.h"

static __global__ void kernel_matrix_multiply(
        const int n, const int m, const int p,
        const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    int i = threadIdx.y + blockDim.y * blockIdx.y; // row
    int j = threadIdx.x + blockDim.x * blockIdx.x; // col

    if (i >= n || j >= m) {
        return;
    }

    float val = 0;

    for (int k = 0; k < (int)p; k++) {
        val += a[i * p + k] * b[k * m + j];
    }

    c[i * m + j] = val;
}

extern "C" void cuda_matrix_multiply(
        size_t n, size_t m, size_t p,
        const float *dev_a, const float *dev_b, float *dev_c,
        cudaStream_t stream) {

    dim3 block_size(128, 1);
    dim3 grid_size(CEIL_DIV(m, block_size.x),
                   CEIL_DIV(n, block_size.y));

    kernel_matrix_multiply<<<grid_size, block_size, 0, stream>>>(
            (int) n, (int) m, (int) p,
            dev_a, dev_b, dev_c);
}
