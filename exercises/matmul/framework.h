#pragma once

#include <stdbool.h>

#include "common.h"
#include "matrix.h"

#define DEVICE_UNDEFINED (0)
#define DEVICE_CPU (1)
#define DEVICE_GPU (2)
#define DEVICE_HYBRID (DEVICE_CPU | DEVICE_GPU)

typedef struct {
    int device;
    size_t size_n;
    size_t size_m;
    size_t size_p;
    bool validate;
    bool verbose;
    double alpha;
    size_t granularity;
} params_t;

typedef void (*callback_t)(
        const params_t *params,
        const matrix_t *a,
        const matrix_t *b,
        matrix_t *c);


EXTERN_C int benchmark_platform(int argc, char *argv[], const char *platform_name,
        callback_t platform_init,
        callback_t platform_compute,
        callback_t platform_deinit);
