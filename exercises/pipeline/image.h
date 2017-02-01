#pragma once

#include "common.h"


typedef struct {
    int width;
    int height;
    unsigned char *data;
} image_t;


EXTERN_C image_t image_allocate(int width, int height);
EXTERN_C void image_free(image_t *img);

EXTERN_C void image_random_fill(image_t *img);

EXTERN_C void image_horz_flip(int width, int height, unsigned char *img);
EXTERN_C void image_find_min_max(int width, int height, const unsigned char *img, int *min, int *max);
EXTERN_C void image_contrast(int width, int height, unsigned char *img, int min, int max);
EXTERN_C void image_smooth(int width, int height, const unsigned char *src, unsigned char *dst);
