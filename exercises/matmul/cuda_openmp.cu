#include <omp.h>

#include "common.h"
#include "framework.h"
#include "cuda_kernel.h"

static cudaEvent_t event_begin;
static cudaEvent_t event_before_kernel;
static cudaEvent_t event_after_kernel;
static cudaEvent_t event_end;

static float *dev_a_data;
static float *dev_b_data;
static float *dev_c_data;

static size_t repeats;
static double time_h2d;
static double time_d2h;
static double time_cpu;
static double time_gpu;


void init(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    CUDA_CHECK(cudaSetDevice, 0);

    CUDA_CHECK(cudaMalloc, &dev_a_data, sizeof(float) * a->cols * a->rows);
    CUDA_CHECK(cudaMalloc, &dev_b_data, sizeof(float) * b->cols * b->rows);
    CUDA_CHECK(cudaMalloc, &dev_c_data, sizeof(float) * c->cols * c->rows);

    // pin memory for fast transfers
    CUDA_CHECK(cudaHostRegister, a->data, sizeof(float) * a->cols * a->rows, 0);
    CUDA_CHECK(cudaHostRegister, b->data, sizeof(float) * b->cols * b->rows, 0);
    CUDA_CHECK(cudaHostRegister, c->data, sizeof(float) * c->cols * c->rows, 0);

    CUDA_CHECK(cudaEventCreate, &event_begin);
    CUDA_CHECK(cudaEventCreate, &event_before_kernel);
    CUDA_CHECK(cudaEventCreate, &event_after_kernel);
    CUDA_CHECK(cudaEventCreate, &event_end);

    repeats = 0;
    time_h2d = 0;
    time_d2h = 0;
    time_cpu = 0;
    time_gpu = 0;
}


void run_cpu_task(size_t row_start, size_t row_end,
        const matrix_t *a,
        const matrix_t *b,
        matrix_t *c) {
    matrix_t ma = matrix_extract_rows((matrix_t*) a, row_start, row_end);
    matrix_t mc = matrix_extract_rows((matrix_t*) c, row_start, row_end);

    matrix_multiply(&ma, b, &mc);
}


void launch_gpu_kernel_async(size_t row_start, size_t row_end,
        const matrix_t *a,
        const matrix_t *b,
        matrix_t *c) {
    size_t num_rows = row_end - row_start;

    CUDA_CHECK(cudaSetDevice, 0);

    CUDA_CHECK(cudaEventRecord, event_begin, CUDA_DEFAULT_STREAM);

    // Copy host -> device
    CUDA_CHECK(cudaMemcpyAsync,
            dev_a_data,
            a->data + row_start * a->cols,
            sizeof(float) * a->cols * num_rows,
            cudaMemcpyHostToDevice,
            CUDA_DEFAULT_STREAM);

    CUDA_CHECK(cudaMemcpyAsync,
            dev_b_data,
            b->data,
            sizeof(float) * b->cols * b->rows,
            cudaMemcpyHostToDevice,
            CUDA_DEFAULT_STREAM);


    // Launch kernel
    CUDA_CHECK(cudaEventRecord, event_before_kernel, CUDA_DEFAULT_STREAM);

    cuda_matrix_multiply(
            num_rows, c->cols, b->rows,
            dev_a_data, dev_b_data, dev_c_data,
            CUDA_DEFAULT_STREAM);

    CUDA_CHECK(cudaEventRecord, event_after_kernel, CUDA_DEFAULT_STREAM);


    // Copy device -> host
    CUDA_CHECK(cudaMemcpyAsync,
            c->data + row_start * c->cols,
            dev_c_data,
            sizeof(float) * c->cols * num_rows,
            cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaEventRecord, event_end, CUDA_DEFAULT_STREAM);
}


void compute(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    size_t n = c->rows;
    size_t rows_cpu, rows_gpu;

    // find number of rows for CPU/GPU
    if (params->device == DEVICE_CPU) {
        rows_cpu = n;
        rows_gpu = 0;
    } else if (params->device == DEVICE_GPU) {
        rows_cpu = 0;
        rows_gpu = n;
    } else {
        rows_cpu = n * params->alpha;
        rows_gpu = n - rows_cpu;
    }

    // launch GPU memory transfers+kernel asynchronously
    launch_gpu_kernel_async(0, rows_gpu, a, b, c);

    // run CPU threads
    stopwatch_t cpu_before = timer();
#pragma omp parallel
    {
        size_t thread_id = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();
        size_t rows_per_thread = CEIL_DIV(rows_cpu, num_threads);

        // find start and end row, first 'rows_gpu' rows are done by GPU
        size_t row_start = rows_gpu + (thread_id + 0) * rows_per_thread;
        size_t row_end   = rows_gpu + (thread_id + 1) * rows_per_thread;

        // prevent overflow if rows_cpu does not divide equally over threads
        row_start = MIN(row_start, n);
        row_end = MIN(row_end, n);

        run_cpu_task(row_start, row_end, a, b, c);
    }
    stopwatch_t cpu_after = timer();

    // wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize);


    // add timing data
    float time1, time2, time3;
    cudaEventElapsedTime(&time1, event_begin, event_before_kernel);
    cudaEventElapsedTime(&time2, event_before_kernel, event_after_kernel);
    cudaEventElapsedTime(&time3, event_after_kernel, event_end);

    repeats++;
    time_h2d += time1 / 1000;
    time_gpu += time2 / 1000;
    time_cpu += timer_diff(cpu_before, cpu_after);
    time_d2h += time3 / 1000;
}


void deinit(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    CUDA_CHECK(cudaFree, dev_a_data);
    CUDA_CHECK(cudaFree, dev_b_data);
    CUDA_CHECK(cudaFree, dev_c_data);

    CUDA_CHECK(cudaEventDestroy, event_begin);
    CUDA_CHECK(cudaEventDestroy, event_before_kernel);
    CUDA_CHECK(cudaEventDestroy, event_after_kernel);
    CUDA_CHECK(cudaEventDestroy, event_end);

    CUDA_CHECK(cudaHostUnregister, a->data);
    CUDA_CHECK(cudaHostUnregister, b->data);
    CUDA_CHECK(cudaHostUnregister, c->data);

    // print timing results if verbose enabled
    if (params->verbose) {
        printf("timing results (avg. over %d runs)\n", (int)repeats);
        printf(" - Host -> device transfer: %f sec\n", time_h2d / repeats);
        printf(" - Device -> host transfer: %f sec\n", time_d2h / repeats);
        printf(" - GPU compute: %f sec\n", time_gpu / repeats);
        printf(" - CPU compute: %f sec\n", time_cpu / repeats);
        printf("\n");
    }
}


int main(int argc, char *argv[]) {
    benchmark_platform(argc, argv,
            "CUDA + OpenMP",
            init,
            compute,
            deinit);
}
