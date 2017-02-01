#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <starpu.h>

#include "common.h"
#include "framework.h"
#include "image.h"
#include "cuda_kernels.h"

/***
 * CPU task: copy src (buffers[0]) to dst (buffers[1])
 */
void task_copy(void *buffers[], void *args) {
    int width = STARPU_MATRIX_GET_NX(buffers[0]);
    int height = STARPU_MATRIX_GET_NY(buffers[0]);
    const unsigned char *src = (unsigned char*) STARPU_MATRIX_GET_PTR(buffers[0]);
    unsigned char *dst = (unsigned char*) STARPU_MATRIX_GET_PTR(buffers[1]);

    memcpy((void*) dst, (const void*) src, width * height * sizeof(unsigned char));
}


/***
 * CPU task: perform horizontal flip of image (buffers[0]).
 */
void task_flip(void *buffers[], void *args) {
    unsigned char *img = (unsigned char*) STARPU_MATRIX_GET_PTR(buffers[0]);
    int width = STARPU_MATRIX_GET_NX(buffers[0]);
    int height = STARPU_MATRIX_GET_NY(buffers[0]);
    image_horz_flip(width, height, img);
}


/***
 * CPU task: calculate min/max (buffers[1]) of image (buffers[0]).
 */
void task_find_min_max(void *buffers[], void *args) {
    int width = STARPU_MATRIX_GET_NX(buffers[0]);
    int height = STARPU_MATRIX_GET_NY(buffers[0]);
    const unsigned char *img = (const unsigned char*) STARPU_MATRIX_GET_PTR(buffers[0]);
    int *min_max = (int*) STARPU_VECTOR_GET_PTR(buffers[1]);
    image_find_min_max(width, height, img, &min_max[0], &min_max[1]);
}


/***
 * CPU task: adjust contrast of image (buffers[0]) using min/max values (buffers[1]).
 */
void task_contrast(void *buffers[], void *args) {
    // TODO:
    // step 1: unpack buffers
    // step 2: call image_contrast(...)
    fprintf(stderr, "error: task_contrast not implemented\n"); exit(1);
}


/***
 * GPU task: flip image (buffers[0]) on horizontal axis.
 */
void gpu_task_flip(void *buffers[], void *args) {
    unsigned char *img = (unsigned char*) STARPU_MATRIX_GET_PTR(buffers[0]);
    int width = STARPU_MATRIX_GET_NX(buffers[0]);
    int height = STARPU_MATRIX_GET_NY(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cuda_launch_horz_flip(stream, width, height, img);
}


/***
 * GPU task: calculate min/max (buffers[1]) of image (buffers[0]).
 */
void gpu_task_find_min_max(void *buffers[], void *args) {
    int width = STARPU_MATRIX_GET_NX(buffers[0]);
    int height = STARPU_MATRIX_GET_NY(buffers[0]);
    const unsigned char *img = (const unsigned char*) STARPU_MATRIX_GET_PTR(buffers[0]);
    int *min_max = (int*) STARPU_VECTOR_GET_PTR(buffers[1]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cuda_launch_find_min_max(stream, width, height, img,
            &min_max[0], &min_max[1]);
}


/***
 * GPU task: adjust contrast of image (buffers[0]) using min/max values (buffers[1]).
 */
void gpu_task_contrast(void *buffers[], void *args) {
    // TODO:
    // step 1: unpack buffers
    // step 2: call cuda_launch_contrast(...)
    fprintf(stderr, "error: gpu_task_contrast not implemented\n"); exit(1);
}



/***
 * Codelet definitions for StarPU
 */
static struct starpu_codelet cl_copy = {
    .where = STARPU_CPU,
    .cpu_func = task_copy,
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W}, // args: src, dst
    .cuda_flags = {STARPU_CUDA_ASYNC}
};

static struct starpu_codelet cl_flip = {
    .where = STARPU_CPU /*| STARPU_CUDA */,
    .cpu_func = task_flip,
    //.cuda_func = gpu_task_flip,
    .nbuffers = 1,
    .modes = {STARPU_RW}, // args: img
    .cuda_flags = {STARPU_CUDA_ASYNC}
};

static struct starpu_codelet cl_find_min_max = {
    .where = STARPU_CPU /*| STARPU_CUDA */,
    .cpu_func = task_find_min_max,
    //.cuda_func = gpu_task_find_min_max,
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W}, // args: img, min/max
    .cuda_flags = {STARPU_CUDA_ASYNC}
};

// TODO: define starpu contrast codelet, should be something like:
/*
static struct starpu_codelet cl_contrast = {
    .where = ?, .cpu_func= ?,
    ...
    .nbuffers = 2,
    .modes = {STARPU_RW, STARPU_R}, // args: img, min/max
};
*/


static starpu_data_handle_t *input_handles;
static starpu_data_handle_t *output_handles;
static starpu_data_handle_t *min_max_handles;

void init(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    input_handles = malloc(sizeof(starpu_data_handle_t) * num_images);
    output_handles = malloc(sizeof(starpu_data_handle_t) * num_images);
    min_max_handles = malloc(sizeof(starpu_data_handle_t) * num_images);

    // initialize starpu
    if (starpu_init(NULL) != 0) {
        fprintf(stderr, "error: initialization of StarPU failed!\n");
        exit(EXIT_FAILURE);
    }

    // pin images for fast data transfers to GPU
    for (int i = 0; i < num_images; i++) {
        size_t size = inputs[i].width * inputs[i].height * sizeof(unsigned char);
        starpu_memory_pin(inputs[i].data, size);
        starpu_memory_pin(outputs[i].data, size);
    }

    // register all handles
    for (int i = 0; i < num_images; i++) {
        int width = inputs[i].width;
        int height = inputs[i].height;


        // program provided buffers
        starpu_matrix_data_register(
                &input_handles[i], STARPU_MAIN_RAM,
                (uintptr_t) inputs[i].data,
                width, width, height,
                sizeof(unsigned char));

        starpu_matrix_data_register(
                &output_handles[i], STARPU_MAIN_RAM,
                (uintptr_t) outputs[i].data,
                width, width, height,
                sizeof(unsigned char));


        // temporary buffers, automatically allocated/freed by StarPU
        starpu_vector_data_register(
                &min_max_handles[i], -1,
                (uintptr_t) 0,
                2,
                sizeof(int));
    }
}



void compute(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {

    // launch all tasks
    for (int i = 0; i < num_images; i++) {
        starpu_data_handle_t img_handle = output_handles[i];

        // copy input image -> output image
        starpu_task_insert(&cl_copy,
                STARPU_R, input_handles[i],
                STARPU_W, img_handle,
                0);

        // launch flip
        if (params->phase_flip) {
            starpu_task_insert(&cl_flip,
                    STARPU_RW, img_handle,
                    0);
        }

        // launch contrast
        if (params->phase_contrast) {
            starpu_task_insert(&cl_find_min_max,
                    STARPU_R, img_handle,
                    STARPU_W, min_max_handles[i],
                    0);

            // TODO: launch contrast codelet
            // should be something like:
            // starpu_task_insert(&cl_contrast, ...);
        }
    }

    // wait for tasks to finish
    starpu_task_wait_for_all();
}


void deinit(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    for (int i = 0; i < num_images; i++) {
        starpu_data_unregister(input_handles[i]);
        starpu_data_unregister(output_handles[i]);
        starpu_data_unregister(min_max_handles[i]);
    }

    for (int i = 0; i < num_images; i++) {
        size_t size = inputs[i].width * inputs[i].height * sizeof(unsigned char);
        starpu_memory_unpin(inputs[i].data, size);
        starpu_memory_unpin(outputs[i].data, size);
    }

    starpu_shutdown();

    free(input_handles);
    free(output_handles);
    free(min_max_handles);
}


int main(int argc, const char *argv[]) {
    return benchmark_platform(argc, argv,
            "StarPU",
            init,
            compute,
            deinit);
}
