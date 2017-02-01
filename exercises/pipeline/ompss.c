#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuda_kernels.h"
#include "common.h"
#include "framework.h"
#include "image.h"


/***
 * CPU task: Copy image of dimensions width x height.
 */
//#pragma omp task in([width * height] src) out([width * height] dst)
void task_copy(int width, int height, const unsigned char *src, unsigned char *dst) {
    memcpy(dst, src, width * height * sizeof(unsigned char));
}

/***
 * CPU task: Flip image of dimensions width x height on vertical axis.
 */
//#pragma omp task inout(???)
void task_flip(int width, int height, unsigned char *img) {
    image_horz_flip(width, height, img);
}

/***
 * CPU task: Compute min/max for image of dimensions width x height.
 */
//#pragma omp task in(???) out (???)
void task_find_min_max(int width, int height, const unsigned char *img, int *min, int *max) {
    image_find_min_max(width, height, img, min, max);
}


/***
 * CPU task: Adjust contrast of image of dimensions width x height.
 */
//#pragma omp task in(???) out (???)
void task_contrast(int width, int height, unsigned char *img, int *min, int *max) {
    image_contrast(width, height, img, *min, *max);
}



/***
 * GPU task: Flip image of dimensions width x height on vertical axis.
 */
//#pragma omp target(cuda) implements(task_flip)
//#pragma omp task inout(???)
void gpu_task_flip(int width, int height, unsigned char *device_img) {
    cudaStream_t stream = nanos_get_kernel_execution_stream();
    cuda_launch_horz_flip(stream, width, height, device_img);
}

// TODO: create these functions.
//void gpu_task_find_min_max(???);
//void gpu_task_contrast(???);


void compute(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    int *min = malloc(sizeof(int) * num_images);
    int *max = malloc(sizeof(int) * num_images);

    for (int i = 0; i < num_images; i++) {
        int width = inputs[i].width;
        int height = inputs[i].height;

        unsigned char *img = outputs[i].data;

        // copy image to output
        task_copy(width, height, inputs[i].data, img);

        // flip image
        if (params->phase_flip) {
            task_flip(width, height, img);
        }

        // adjust contrast
        if (params->phase_contrast) {
            // TODO: call find_min_max and contrast
            // should be something like
            // task_find_min_max(..., &min[i], &max[i]);
            // task_contrast(...);
        }
    }

#pragma omp taskwait

    free(min);
    free(max);
}

int main(int argc, const char *argv[]) {
    return benchmark_platform(argc, argv,
            "OmpSs",
            NULL,
            compute,
            NULL);
}
