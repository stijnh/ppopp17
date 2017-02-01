#include <stdlib.h>
#include <stdio.h>

#include "common.h"
#include "framework.h"
#include "cuda_kernels.h"

#define NUM_STREAMS (16)

static unsigned char *dev_images[NUM_STREAMS];
static int *dev_min_maxs[NUM_STREAMS];
cudaStream_t streams[NUM_STREAMS];


void init(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    CUDA_CHECK(cudaSetDevice, 0);

    size_t max_size = 0;
    for (int i = 0; i < num_images; i++) {
        size_t size = inputs[i].width * inputs[i].height * sizeof(unsigned char);

        // pin host memory for fast CPU<->GPU transfers
        CUDA_CHECK(cudaHostRegister, inputs[i].data, size, 0);
        CUDA_CHECK(cudaHostRegister, outputs[i].data, size, 0);

        max_size = MAX(max_size, size);
    }

    // allocate resources for streams
    for (int i = 0; i < MIN(num_images, NUM_STREAMS); i++) {
        CUDA_CHECK(cudaMalloc, &dev_images[i], max_size);
        CUDA_CHECK(cudaMalloc, &dev_min_maxs[i], 2 * sizeof(int));
        CUDA_CHECK(cudaStreamCreate, &streams[i]);
    }
}


void compute(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    CUDA_CHECK(cudaSetDevice, 0);

    for (int i = 0; i < num_images; i++) {
        int width = inputs[i].width;
        int height = inputs[i].height;

        // images are assigned to streams in round robin fashion
        size_t stream_index = i % NUM_STREAMS;
        cudaStream_t stream = streams[stream_index];
        unsigned char *dev_image = dev_images[stream_index];
        int *dev_min_max = dev_min_maxs[stream_index];

        CUDA_CHECK(cudaMemcpyAsync,
                dev_image,
                inputs[i].data,
                width * height * sizeof(unsigned char),
                cudaMemcpyHostToDevice,
                stream);

        // flip image
        if (params->phase_flip) {
            cuda_launch_horz_flip(
                    stream,
                    width,
                    height,
                    dev_image);
        }

        // contrast adjustment
        if (params->phase_contrast) {
            cuda_launch_find_min_max(
                    stream,
                    width,
                    height,
                    dev_image,
                    &dev_min_max[0],
                    &dev_min_max[1]);

            cuda_launch_contrast(
                    stream,
                    width,
                    height,
                    dev_image,
                    &dev_min_max[0],
                    &dev_min_max[1]);
        }

        CUDA_CHECK(cudaMemcpyAsync,
                outputs[i].data,
                dev_image,
                width * height * sizeof(unsigned char),
                cudaMemcpyDeviceToHost,
                stream);
    }

    // wait for device to finish
    CUDA_CHECK(cudaDeviceSynchronize);
}


void deinit(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    CUDA_CHECK(cudaSetDevice, 0);

    for (int i = 0; i < num_images; i++) {
        CUDA_CHECK(cudaHostUnregister, inputs[i].data);
        CUDA_CHECK(cudaHostUnregister, outputs[i].data);
    }

    for (int i = 0; i < MIN(num_images, NUM_STREAMS); i++) {
        CUDA_CHECK(cudaFree, dev_images[i]);
        CUDA_CHECK(cudaFree, dev_min_maxs[i]);
        CUDA_CHECK(cudaStreamDestroy, streams[i]);
    }
}


int main(int argc, const char *argv[]) {
    return benchmark_platform(argc, argv,
            "CUDA",
            init,
            compute,
            deinit);
}
