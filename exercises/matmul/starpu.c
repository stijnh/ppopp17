#include <starpu.h>

#include "framework.h"
#include "cuda_kernel.h"

// forward declarations
void task_matrix_multiply_cpu(void *buffers[], void *args);
void task_matrix_multiply_gpu(void *buffers[], void *args);


static struct starpu_perfmodel perf_model = {
    .symbol = "matrix multiply",
    .type = STARPU_HISTORY_BASED
};


static struct starpu_codelet global_cl = {
    .where = 0, // dynamically set in compute(...)
    .cpu_func = task_matrix_multiply_cpu,
    .cuda_func = task_matrix_multiply_gpu,
    .nbuffers = 3,
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    //.model = &perf_model /* Uncomment to enable history-based model */
};


// task definition for GPU
void task_matrix_multiply_gpu(void *buffers[], void *args) {
    float *a = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
    float *b = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
    float *c = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);

    size_t n = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t m = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t p = STARPU_MATRIX_GET_NX(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cuda_matrix_multiply(n, m, p, a, b, c, stream);
}


// task definition for CPU
void task_matrix_multiply_cpu(void *buffers[], void *args) {
    float *a = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
    float *b = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
    float *c = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);

    size_t n = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t m = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t p = STARPU_MATRIX_GET_NX(buffers[0]);

    matrix_t ma = matrix_create(n, p, a);
    matrix_t mb = matrix_create(p, m, b);
    matrix_t mc = matrix_create(n, m, c);

    matrix_multiply(&ma, &mb, &mc);
}


void init(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    if (starpu_init(NULL) != 0) {
        fprintf(stderr, "error: initialization of StarPU failed\n");
        exit(EXIT_FAILURE);
    }

    // pin host memory for fast CPU<->GPU data transfers.
    starpu_memory_pin(a->data, a->cols * a->rows * sizeof(float));
    starpu_memory_pin(b->data, b->cols * b->rows * sizeof(float));
    starpu_memory_pin(c->data, c->cols * c->rows * sizeof(float));
}


void compute(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    size_t m = a->rows;
    size_t rows_per_task = params->granularity;
    size_t num_tasks = CEIL_DIV(m, rows_per_task);

    struct starpu_codelet cl = global_cl;
    if      (params->device == DEVICE_CPU) cl.where = STARPU_CPU;
    else if (params->device == DEVICE_GPU) cl.where = STARPU_CUDA;
    else                                   cl.where = STARPU_CUDA | STARPU_CPU;

    starpu_data_handle_t a_handle;
    starpu_data_handle_t b_handle;
    starpu_data_handle_t c_handle;

    // register memory for StarPU run-time system
    starpu_matrix_data_register(
            &a_handle, STARPU_MAIN_RAM,
            (uintptr_t) a->data,
            a->cols, a->cols, a->rows,
            sizeof(float));

    starpu_matrix_data_register(
            &b_handle, STARPU_MAIN_RAM,
            (uintptr_t) b->data,
            b->cols, b->cols, b->rows,
            sizeof(float));

    starpu_matrix_data_register(
            &c_handle, STARPU_MAIN_RAM,
            (uintptr_t) c->data,
            c->cols, c->cols, c->rows,
            sizeof(float));

    // split matrix A and C vertically into blocks
    struct starpu_data_filter filter = {
        .filter_func = starpu_matrix_filter_vertical_block,
        .nchildren = num_tasks,
    };

    starpu_data_partition(a_handle, &filter);
    starpu_data_partition(c_handle, &filter);

    for (size_t i = 0; i < num_tasks; i++) {
        starpu_task_insert(&cl,
                STARPU_R, starpu_data_get_sub_data(a_handle, 1, i),
                STARPU_R, b_handle,
                STARPU_W, starpu_data_get_sub_data(c_handle, 1, i),
                0);
    }

    starpu_task_wait_for_all();

    starpu_data_unpartition(a_handle, STARPU_MAIN_RAM);
    starpu_data_unpartition(c_handle, STARPU_MAIN_RAM);

    starpu_data_unregister(a_handle);
    starpu_data_unregister(b_handle);
    starpu_data_unregister(c_handle);
}


void deinit(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    starpu_memory_unpin(a->data, a->cols * a->rows * sizeof(float));
    starpu_memory_unpin(b->data, b->cols * b->rows * sizeof(float));
    starpu_memory_unpin(c->data, c->cols * c->rows * sizeof(float));

    starpu_shutdown();
}


int main(int argc, char *argv[]) {
    return benchmark_platform(argc, argv,
            "StarPU",
            init,
            compute,
            deinit);
}
