#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
#include "framework.h"
#include "image.h"


void compute(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    for (int i = 0; i < num_images; i++) {
        int w = inputs[i].width;
        int h = inputs[i].height;
        unsigned char *img = outputs[i].data;


        // copy input to outputs
        memcpy((void*) img, (const void*) inputs[i].data,
                sizeof(unsigned char) * w * h);

        // flip image
        if (params->phase_flip) {
            image_horz_flip(w, h, img);
        }

        // adjust contrast
        if (params->phase_contrast) {
            int min, max;
            image_find_min_max(w, h, img, &min, &max);
            image_contrast(w, h, img, min, max);
        }

        // perform smoothing
        if (params->phase_smooth) {
            unsigned char *tmp = malloc(w * h * sizeof(unsigned char));

            image_smooth(w, h, img, tmp);
            memcpy((void*) img, (const void*) tmp,
                    sizeof(unsigned char) * w * h);

            free(tmp);
        }
    }
}

int main(int argc, const char *argv[]) {
    return benchmark_platform(argc, argv,
            "sequential baseline",
            NULL,
            compute,
            NULL);
}
