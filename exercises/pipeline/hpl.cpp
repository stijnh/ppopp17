#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <HPL.h>

#include "common.h"
#include "framework.h"
#include "image.h"

using namespace HPL;


void init(const params_t *params, int num_images, image_t *inputs, image_t *outputs);
void compute(const params_t *params, int num_images, image_t *inputs, image_t *outputs);
void deinit(const params_t *params, int num_images, image_t *inputs, image_t *outputs);

/**
 * Three kernels of image pipeline.
 */
void kernel_horz_flip(
        Array<unsigned char, 1> img,
        Int width) {
    Int x, y;
    get_global_id(x, y);

    Int rev_x = width - x - 1;
    Array<unsigned char, 0> tmp;

    if_ (x < width / 2) {
        tmp = img[y * width + x];
        img[y * width + x] = img[y * width + rev_x];
        img[y * width + rev_x] = tmp;
    }
}

void kernel_find_min_max(
        Array<unsigned char, 1> img,
        Array<int, 1> lower_upper,
        Int width) {
    Int x, y;
    get_global_id(x, y);

    Array<unsigned char, 0> pixel = img[y * width + x];

    atomic_min(lower_upper[0], pixel);
    atomic_max(lower_upper[1], pixel);
}

void kernel_contrast(
        Array<unsigned char, 1> img,
        Array<int, 1> lower_upper,
        Int width) {
    Int x, y;
    get_global_id(x, y);

    Int lower = lower_upper[0];
    Int upper = lower_upper[1];
    Array<unsigned char, 0> pixel = img[y * width + x];

    // TODO: fill in the rest
    fprintf(stderr, "error: kernel_contrast not implemented\n"); exit(1);
}


/**
 * Three partitioner for execution plan definitions.
 */
void flip_partitioner(
        FRunner &fr, Range ranges[3],
        Array<unsigned char, 1> &img,
        Int &width) {
    size_t w = width.value();
    size_t start = ranges[0].origin * w;
    size_t end = (ranges[0].end + 1) * w - 1;

    fr(img(Range(start, end)), width);
}

void find_min_max_partitioner(
        FRunner &fr, Range ranges[3],
        Array<unsigned char, 1> &img,
        Array<int, 1> &lower_upper,
        Int &width) {
    int offset = fr.getDevice().getDeviceType() == CPU ? 0 : 2;

    size_t w = width.value();
    size_t start = ranges[0].origin * w;
    size_t end = (ranges[0].end + 1) * w - 1;

    fr(img(Range(start, end)), lower_upper(Range(offset, offset + 1)), width);
}

void contrast_partitioner(
        FRunner &fr, Range ranges[3],
        Array<unsigned char, 1> &img,
        Array<int, 1> &lower_upper,
        Int &width) {

    size_t w = width.value();
    size_t start = ranges[0].origin * w;
    size_t end = (ranges[0].end + 1) * w - 1;

    fr(img(Range(start, end)), lower_upper(Range(0,1)), width);
}


// Execution plan definitions
static ExecutionPlan flip_plan(flip_partitioner,                 HPL::ADAPT2);
static ExecutionPlan find_min_max_plan(find_min_max_partitioner, HPL::ADAPT2);
static ExecutionPlan contrast_plan(contrast_partitioner,         HPL::ADAPT2);


void init(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    ExecutionPlan *plans[] = {
        &flip_plan,
        &find_min_max_plan,
        &contrast_plan
    };

    for (size_t i = 0; i < 3; i++) {
        plans[i]->add(Device(CPU), 50);
        plans[i]->add(Device(GPU), 50);
    }

    // call compute for first image to tune execution_plan.
    compute(params, 1, inputs, outputs);
}


void compute(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {

    // find largest image size.
    size_t max_size = 0;
    for (int i = 0; i < num_images; i++) {
        max_size = MAX(max_size, inputs[i].width * inputs[i].height);
    }


    // allocate memory
    Array<int, 1> lower_upper(4, HPL_FAST_MEM);
    Array<unsigned char, 1> image(max_size, HPL_FAST_MEM);


    for (int i = 0; i < num_images; i++) {
        int width = inputs[i].width;
        int height = inputs[i].height;
        Int width_arg = width;

        // copy to pointer
        memcpy((void*) image.data(HPL_WR),
                (const void*)inputs[i].data,
                width * height * sizeof(unsigned char));

        // execute flip
        if (params->phase_flip) {
            eval(kernel_horz_flip)
                .local(8,8)
                .global(height, width)
                .executionPlan(flip_plan)
                (image, width_arg);
        }

        // execute contrast
        if (params->phase_contrast) {

            // set initial min/max values
            int *lu_ptr = lower_upper.data(HPL_WR);
            lu_ptr[0] = lu_ptr[2] = 255;
            lu_ptr[1] = lu_ptr[3] = 0;

            // launch min/max kernel
            eval(kernel_find_min_max)
                .global(height, width)
                .executionPlan(find_min_max_plan)
                (image, lower_upper, width_arg);

            // lower_upper contains 4 entries:
            // [CPU_min, CPU_max, GPU_min, GPU_max]
            // reduce values into first two entries.
            lu_ptr = lower_upper.data(HPL_RDWR);
            lu_ptr[0] = MIN(lu_ptr[0], lu_ptr[2]);
            lu_ptr[1] = MAX(lu_ptr[1], lu_ptr[3]);

            // launch contrast kernel
            eval(kernel_contrast)
                .global(height, width)
                .executionPlan(contrast_plan)
                (image, lower_upper, width_arg);
        }

        memcpy((void*) outputs[i].data,
                (const void*) image.data(HPL_RD),
                width * height * sizeof(unsigned char));
    }
}


void deinit(const params_t *params, int num_images, image_t *inputs, image_t *outputs) {
    auto weights1 = flip_plan.getWeights();
    auto weights2 = find_min_max_plan.getWeights();
    auto weights3 = contrast_plan.getWeights();

    printf("final weights (CPU vs GPU):\n");
    printf(" - flip:         %d%% vs %d%%\n", weights1[0], weights1[1]);
    printf(" - find min/max: %d%% vs %d%%\n", weights2[0], weights2[1]);
    printf(" - contrast:     %d%% vs %d%%\n", weights3[0], weights3[1]);
}


int main(int argc, const char *argv[]) {
    return benchmark_platform(argc, argv,
            "HPL",
            init,
            compute,
            deinit);
}
