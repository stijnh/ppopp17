#pragma once

#include <stdbool.h>

#include "image.h"
#include "common.h"

typedef struct {
    int num_images;
    int size;
    bool phase_flip;
    bool phase_smooth;
    bool phase_contrast;
    bool validate;
} params_t;

typedef void (*callback_t)(
        const params_t *params,
        int num_images,
        image_t *inputs,
        image_t *output);

EXTERN_C int benchmark_platform(int argc, const char *argv[],
        const char *platform_name,
        callback_t platform_init,
        callback_t platform_compute,
        callback_t platform_deinit);
