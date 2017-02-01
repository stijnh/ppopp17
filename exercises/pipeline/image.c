#include <stdio.h>
#include <stdlib.h>

#include "image.h"


image_t image_allocate(int width, int height) {
    unsigned char *data = malloc(sizeof(unsigned char) * width * height);

    image_t img;
    img.width = width;
    img.height = height;
    img.data = data;
    return img;
}


void image_free(image_t *img) {
    free(img->data);
    img->width = 0;
    img->height = 0;
}


void image_random_fill(image_t *img) {
    unsigned int seed = rand();
    int lower = rand() % 64 + 8;           // all pixels are >= lower
    int upper = 255 - (rand() % 64 + 8);   // all pixels are < upper

    for (int i = 0; i < img->width * img->height; i++) {
        img->data[i] = rand_r(&seed) % (upper - lower) + lower;
    }
}

void image_horz_flip(int width, int height, unsigned char *img) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width / 2; x++) {
            SWAP(img[y * width + x],
                 img[y * width + (width - x - 1)]);
        }
    }
}


void image_find_min_max(int width, int height, const unsigned char *img, int *min_arg, int *max_arg) {
    int min = img[0];
    int max = img[0];

    for (int i = 0; i < width * height; i++) {
        min = MIN(min, (int) img[i]);
        max = MAX(max, (int) img[i]);
    }

    *min_arg = min;
    *max_arg = max;
}


void image_contrast(int width, int height, unsigned char *img, int min, int max) {
    double lower = 0.9 * min + 0.1 * max;
    double upper = 0.1 * min + 0.9 * max;

    for (int i = 0; i < width * height; i++) {
        unsigned char old_pixel = img[i];
        unsigned char new_pixel;

        if (old_pixel <= lower) {
            new_pixel = 0;
        } else if (old_pixel >= upper) {
            new_pixel = 255;
        } else {
            new_pixel = (int)(255.0 * (old_pixel - lower) / (upper - lower));
        }

        img[i] = new_pixel;
    }
}

void image_smooth(int width, int height, const unsigned char *src, unsigned char *dst) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;

            for (int dx = x - 1; dx <= x + 1; dx++) {
                for (int dy = y - 1; dy <= y + 1; dy++) {
                    if (dx >= 0 && dx < width && dy >= 0 && dy < height) {
                        sum += src[dy * width + dx];
                    }
                }
            }

            int average = sum / 9;
            dst[y * width + x] = average;
        }
    }
}
