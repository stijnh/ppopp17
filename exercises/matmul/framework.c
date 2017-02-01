#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "framework.h"

#define MIN_COMPUTE_TIME (10.0f)
#define MIN_REPEATS (3)
#define MAX_REPEATS (1000)


static bool validate_matrix_mult(const matrix_t *a, const matrix_t *b, matrix_t *c) {
    assert(a->rows == c->rows);
    assert(a->cols == b->rows);
    assert(b->cols == c->cols);

    size_t n = c->rows;
    size_t m = c->cols;
    size_t p = b->rows;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++){
            float ref = c->data[i * c->cols + j];
            float val = 0;

            for (size_t k = 0; k < p; k++){
                val += a->data[i * p + k] * b->data[k * m + j];
            }

            if (isnan(ref) || isinf(ref) || fabs(ref - val) > 0.001f) {
                fprintf(stderr, "error: entry (%d, %d) is incorrect, "
                        "found %f but expected %f.\n",
                        (int)i, (int)j, ref, val);
                return false;
            }
        }
    }

    return true;
}


static void print_help(int argc, char *argv[]) {
    printf("usage: %s SIZE [DEVICE] [OPTION]...\n", argv[0]);
    printf("Performs matrix multiplication for dimensions SIZE x SIZE.\n");
    printf("-h         Print help message.\n");
    printf("-s         Skip validation stage.\n");
    printf("-a [ALPHA] Set value of alpha for static workload distribution.\n");
    printf("-g [SIZE]  Set rows-per-task parameter for jobs.\n");
    printf("-v         Enable verbose output.\n");
}


static bool parse_long(const char *str, long long int *size) {
    char *end;
    long long int result = strtoll(str, &end, 10);

    if (str != end) {
        *size = result;
        return true;
    }

    return false;
}


static bool parse_args(int argc, char *argv[], params_t *params) {
    if (argc < 2) {
        print_help(argc, argv);
        return false;
    }

    params->device = DEVICE_HYBRID;
    params->verbose = false;
    params->validate = true;
    params->alpha = 0.5;
    params->granularity = 16;

    // parse n
    int arg_offset = 1;
    if (!parse_long(argv[arg_offset], (long long int*)&(params->size_n))) {
        print_help(argc, argv);
        return false;
    }

    arg_offset++;
    params->size_m = params->size_n;
    params->size_p = params->size_n;

    // parse m
    if (arg_offset < argc && parse_long(argv[arg_offset], (long long int*)&(params->size_m))) {
        arg_offset++;
    }

    // parse p
    if (arg_offset < argc && parse_long(argv[arg_offset], (long long int*)&(params->size_p))) {
        arg_offset++;
    }

    if (arg_offset < argc) {
        const char *dev = argv[arg_offset++];

        if (stricmp(dev, "cpu") == 0) {
            params->device = DEVICE_CPU;
            params->alpha = 1;
        } else if (stricmp(dev, "gpu") == 0) {
            params->device = DEVICE_GPU;
            params->alpha = 0;
        } else if (stricmp(dev, "hybrid") == 0 || stricmp(dev, "hyb") == 0) {
            params->device = DEVICE_HYBRID;
            params->alpha = 0.5;
        } else if (dev[0] == '-') {
            arg_offset--; // go to prev arg
        } else {
            fprintf(stderr, "error: unknown device %s.\n", dev);
            return false;
        }
    }

    while (arg_offset < argc) {
        const char *arg = argv[arg_offset++];

        if (strcmp(arg, "-s") == 0) {
            params->validate = false;
        }

        else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_help(argc, argv);
            return false;
        }

        else if (strcmp(arg, "-a") == 0) {
            if (arg_offset >= argc) {
                print_help(argc, argv);
                return false;
            }

            if (params->device != DEVICE_HYBRID) {
                fprintf(stderr, "error: option '-a' can only be used "
                        "for hybrid platform.\n");
                return false;
            }

            params->alpha = atof(argv[arg_offset++]);
        }

        else if (strcmp(arg, "-g") == 0) {
            if (arg_offset >= argc) {
                print_help(argc, argv);
                return false;
            }

            if (!parse_long(argv[arg_offset++], (long long int*)&params->granularity)) {
                fprintf(stderr, "error: invalid granularity.\n");
                return false;
            }
        }

        else if (strcmp(arg, "-v") == 0) {
            params->verbose = true;
        }

        else {
            fprintf(stderr, "error: unknown option '%s', try --help.\n", arg);
            return false;
        }
    }

    if (params->size_n == 0 || params->size_m == 0 || params->size_p == 0) {
        fprintf(stderr, "error: matrix size should be > 0\n");
        return false;
    }

    if (params->validate && MAX(params->size_n, MAX(params->size_m, params->size_p)) > 1024) {
        fprintf(stderr, "warning: output validation disable for large matrices.\n");
        params->validate = false;
    }

    if (params->granularity == 0) {
        fprintf(stderr, "error: granularity must be > 0.\n");
        return false;
    }

    if (params->alpha < 0 || params->alpha > 1) {
        fprintf(stderr, "error: alpha must be in range [0, 1].\n");
        return false;
    }

    return true;
}


int benchmark_platform(int argc, char *argv[], const char *platform_name,
        callback_t platform_init, callback_t platform_compute, callback_t platform_deinit) {
    params_t params;

    if (!parse_args(argc, argv, &params)) {
        return EXIT_FAILURE;
    }

    printf("parameters:\n");
    printf(" - matrix A: %zu x %zu\n", params.size_n, params.size_p);
    printf(" - matrix B: %zu x %zu\n", params.size_p, params.size_m);
    printf(" - matrix C: %zu x %zu\n", params.size_n, params.size_m);
    printf(" - device: %s\n",
            params.device == DEVICE_CPU ? "CPU" :
            params.device == DEVICE_GPU ? "GPU" :
            params.device == DEVICE_HYBRID ? "CPU+GPU" :
            "?");
    printf(" - validate output: %s\n", params.validate ? "yes" : "no");
    printf(" - granularity: %zu\n", params.granularity);
    if (params.device == DEVICE_HYBRID) printf(" - alpha: %f\n", params.alpha);
    printf("\n");

    matrix_t a = matrix_allocate(params.size_n, params.size_p);
    matrix_t b = matrix_allocate(params.size_p, params.size_m);
    matrix_t c = matrix_allocate(params.size_n, params.size_m);

    matrix_fill_random(&a);
    matrix_fill_random(&b);


    // perform initialization
    stopwatch_t before_init = timer();
    if (platform_init) {
        platform_init(&params, &a, &b, &c);
    }
    stopwatch_t after_init = timer();


    // perform computation
    size_t repeats = 0;
    stopwatch_t before_comp = timer();
    stopwatch_t after_comp;
    while (1) {
        platform_compute(&params, &a, &b, &c);

        repeats++;
        after_comp = timer();

        if (repeats < MIN_REPEATS) {
            continue;
        }

        if (repeats >= MAX_REPEATS) {
            break;
        }

        if (timer_diff(before_comp, after_comp) > MIN_COMPUTE_TIME) {
            break;
        }
    }


    // perform deinit
    stopwatch_t before_deinit = timer();
    if (platform_deinit) {
        platform_deinit(&params, &a, &b, &c);
    }
    stopwatch_t after_deinit = timer();


    // validate if needed
    bool success = params.validate && validate_matrix_mult(&a, &b, &c);


    // print results!
    printf("benchmark complete: %s\n", platform_name);

    if (!params.validate) {
        printf(" - validation: <skipped>\n");
    } else if (success) {
        printf(" - validation: success!\n");
    } else {
        printf(" - validation: ERROR!\n");
    }

    printf(" - initialization: %.5f sec\n",
            timer_diff(before_init, after_init));
    printf(" - computation: %.5f sec (avg. over %d runs)\n",
            timer_diff(before_comp, after_comp) / repeats, (int)repeats);
    printf(" - deinitialization: %.5f sec \n",
            timer_diff(before_deinit, after_deinit));


    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c);

    return EXIT_SUCCESS;
}
