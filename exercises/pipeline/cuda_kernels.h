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

EXTERN_C void cuda_launch_horz_flip(
        cudaStream_t stream,
        int width, int height,
        unsigned char *dev_data);

EXTERN_C void cuda_launch_find_min_max(
        cudaStream_t stream,
        int width, int height,
        const unsigned char *dev_data,
        int *dev_min,
        int *dev_max);

EXTERN_C void cuda_launch_smooth(
        cudaStream_t stream,
        int width, int height,
        const unsigned char *dev_input,
        unsigned char *dev_output);

EXTERN_C void cuda_launch_contrast(
        cudaStream_t stream,
        int width, int height,
        unsigned char *dev_data,
        int *dev_min,
        int *dev_max);
