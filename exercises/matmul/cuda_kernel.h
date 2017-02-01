#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define CUDA_DEFAULT_STREAM (0)

#ifdef __CUDACC__
#define CUDA_CHECK(f, ...) \
    check_cuda_error_code(f(__VA_ARGS__), #f, __FILE__, __LINE__)

static void check_cuda_error_code(cudaError_t error, const char *fun, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "error: %s:%d: %s: %s\n", file, line, fun, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#endif


EXTERN_C void cuda_matrix_multiply(
        size_t n, size_t m, size_t p,
        const float *dev_a, const float *dev_b, float *dev_c,
        cudaStream_t stream);
