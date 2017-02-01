#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

#include "framework.h"

static bool validate_pipeline(const params_t *params, const image_t *input, image_t *output) {
    int w = input->width;
    int h = input->height;
    unsigned char *result = malloc(w * h);
    unsigned char *scratch = malloc(w * h);

    memcpy((void*) result, (const void*) input->data, w * h);

    if (params->phase_flip) {
        image_horz_flip(w, h, result);
    }

    if (params->phase_contrast) {
        int min, max;
        image_find_min_max(w, h, result, &min, &max);
        image_contrast(w, h, result, min, max);
    }

    if (params->phase_smooth) {
        image_smooth(w, h, result, scratch);
        memcpy((void*) result, (const void*) scratch, w * h);
    }

    bool is_match = true;
    for (int y = 0; y < h && is_match; y++) {
        for (int x = 0; x < w && is_match; x++) {
            size_t idx = y * w + x;

            if (output->data[idx] != result[idx]) {
                fprintf(stderr, "warning: pixel at (%d, %d) is incorrect, "
                        "found value %d but should be value %d.\n",
                        x, y,
                        (int) output->data[idx],
                        (int) result[idx]);

                is_match = false;
            }
        }
    }

    free(scratch);
    free(result);

    return is_match;
}


static void print_help(int argc, const char *argv[]) {
    printf("usage: %s SIZE NUM [OPTION]...\n", argv[0]);
    printf("Create pipeline processing NUM images of dimensions SIZE x SIZE\n");
    printf("-h              Print help message.\n");
    printf("-s              Skip validation stage.\n");
    printf("--do-flip       Perform horizontal flip.\n");
    printf("--do-contrast   Perform contrast adjustment\n");
    printf("--do-smooth     Perform image smoothing\n");
}


static bool parse_args(int argc, const char *argv[], params_t *params) {
    if (argc < 3) {
        print_help(argc, argv);
        return false;
    }

    params->phase_flip = false;
    params->phase_contrast = false;
    params->phase_smooth = false;
    params->validate = true;

    params->size = atoi(argv[1]);
    params->num_images = atoi(argv[2]);

    if (params->num_images <= 0) {
        fprintf(stderr, "error: number of images should be > 0.\n");
        return false;
    }

    if (params->size <= 0) {
        fprintf(stderr, "error: size of images should be > 0.\n");
        return false;
    }

    size_t square = (size_t) params->size *
                    (size_t) params->size;

    if (square > INT_MAX) {
        size_t max_size = 1;
        while (max_size * max_size < INT_MAX) {
            max_size++;
        }

        fprintf(stderr, "error: size of images should be < %d\n", (int)max_size);
        return false;
    }

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--do-flip") == 0) {
            params->phase_flip = true;
        } else if (strcmp(argv[i], "--do-contrast") == 0) {
            params->phase_contrast = true;
        } else if (strcmp(argv[i], "--do-smooth") == 0) {
            params->phase_smooth = true;
        } else if (strcmp(argv[i], "-s") == 0) {
            params->validate = false;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argc, argv);
            return false;
        } else {
            fprintf(stderr, "error: unknown parameter '%s', try --help\n", argv[i]);
            return false;
        }
    }

    if (params->size > 1024 && params->validate) {
        fprintf(stderr, "warning: validation disabled for large images.\n");
        params->validate = false;
    }

    if (!params->phase_flip && !params->phase_smooth && !params->phase_contrast) {
        fprintf(stderr, "error: no phase of the pipeline is active.\n");
        fprintf(stderr, "info: enable a phase using: --do-flip, --do-contrast, or --do-smooth.\n");
        return false;
    }

    size_t memory_per_image = params->size * (size_t) params->size;

    printf("parameters:\n");
    printf(" - number of images: %d\n", params->num_images);
    printf(" - size of images: %d x %d (%.4f MB)\n", params->size, params->size,
            memory_per_image / 1024.0 / 1024.0);
    printf(" - total memory usage: %.4f MB\n",
            2 * (params->num_images * memory_per_image) / 1024.0 / 1024.0);
    printf(" - perform horizontal flip: %s\n", params->phase_flip ? "yes" : "no");
    printf(" - perform smoothing: %s\n", params->phase_smooth ? "yes" : "no");
    printf(" - perform contrast adjustment: %s\n", params->phase_contrast ? "yes" : "no");
    printf(" - validate output: %s\n", params->validate ? "yes" : "no");
    printf("\n");

    return true;
}


int benchmark_platform(int argc, const char *argv[], const char *platform_name,
            callback_t platform_init, callback_t platform_compute, callback_t platform_deinit) {
    params_t params;

    if (!parse_args(argc, argv, &params)) {
        return EXIT_FAILURE;
    }

    int num_images = params.num_images;
    int size = params.size;

    image_t *inputs = malloc(sizeof(image_t) * num_images);
    image_t *outputs = malloc(sizeof(image_t) * num_images);

    for (int i = 0; i < num_images; i++) {
        inputs[i] = image_allocate(size, size);
        outputs[i] = image_allocate(size, size);

        // generating images is slow. only generate the first 10 images,
        // the rest is just copied from this initial set.
        if (i < 10) {
            image_random_fill(&inputs[i]);
        } else {
            memcpy((void*) inputs[i].data,
                    (const void*) inputs[i % 10].data,
                    size * size * sizeof(unsigned char));
        }
    }


    stopwatch_t before_init = timer();
    if (platform_init) {
        platform_init(&params, num_images, inputs, outputs);
    }
    stopwatch_t after_init = timer();

    stopwatch_t before_comp = timer();
    platform_compute(&params, num_images, inputs, outputs);
    stopwatch_t after_comp = timer();

    stopwatch_t before_deinit = timer();
    if (platform_deinit) {
        platform_deinit(&params, num_images, inputs, outputs);
    }
    stopwatch_t after_deinit = timer();


    // only validate first image
    bool success = params.validate && validate_pipeline(&params, &inputs[0], &outputs[0]);


    printf("benchmark complete: %s\n", platform_name);

    if (!params.validate) {
        printf(" - validate: <skipped>\n");
    } else if (success) {
        printf(" - validate: success!\n");
    } else {
        printf(" - validate: ERROR\n");
    }

    printf(" - initialization: %.5f sec\n",
            timer_diff(before_init, after_init));
    printf(" - computation: %.5f sec per image (%.5f in total)\n",
            timer_diff(before_comp, after_comp) / num_images,
            timer_diff(before_comp, after_comp));
    printf(" - deinitialization: %.5f sec \n",
            timer_diff(before_deinit, after_deinit));


    for (int i = 0; i < num_images; i++) {
        image_free(&inputs[i]);
        image_free(&outputs[i]);
    }

    free(inputs);
    free(outputs);

    return EXIT_SUCCESS;
}

