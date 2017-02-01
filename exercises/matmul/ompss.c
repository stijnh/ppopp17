#include "matrix.h"
#include "cuda_kernel.h"

#include "framework.h"


// CPU task
#pragma omp target device(smp) no_copy_deps
#pragma omp task in([num_rows * p] a) in([p * m] b) out([num_rows * m] c)
void task_submatrix_multiply_cpu(
        size_t num_rows, size_t m, size_t p,
        float *a, float *b, float *c) {
    matrix_t ma = matrix_create(num_rows, p, a);
    matrix_t mb = matrix_create(p, m, b);
    matrix_t mc = matrix_create(num_rows, m, c);

    matrix_multiply(&ma, &mb, &mc);
}


// GPU task
#pragma omp target device(cuda)
#pragma omp task in([num_rows * p] a) in([p * m] b) out([num_rows * m] c)
void task_submatrix_multiply_gpu(
        size_t num_rows, size_t m, size_t p,
        float *a, float *b, float *c) {
    cudaStream_t stream = nanos_get_kernel_execution_stream();
    cuda_matrix_multiply(
            num_rows, m, p,
            a, b, c,
            stream);
}


// Hybrid task (CPU implementation)
#pragma omp target device(smp)
#pragma omp task in([num_rows * p] a) in([p * m] b) out([num_rows * m] c)
void task_submatrix_multiply_hybrid(
        size_t num_rows, size_t m, size_t p,
        float *a, float *b, float *c) {
    matrix_t ma = matrix_create(num_rows, p, a);
    matrix_t mb = matrix_create(p, m, b);
    matrix_t mc = matrix_create(num_rows, m, c);

    matrix_multiply(&ma, &mb, &mc);
}


// Hybrid task (GPU implementation)
#pragma omp target device(cuda) implements(task_submatrix_multiply_hybrid)
#pragma omp task in([num_rows * p] a) in([p * m] b) out([num_rows * m] c)
void task_submatrix_multiply_hybrid_gpu(
        size_t num_rows, size_t m, size_t p,
        float *a, float *b, float *c) {
    cudaStream_t stream = nanos_get_kernel_execution_stream();
    cuda_matrix_multiply(
            num_rows, m, p,
            a, b, c,
            stream);
}


void compute(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    size_t n = c->rows;
    size_t m = c->cols;
    size_t p = b->rows;

    int device = params->device;
    size_t rows_per_task = params->granularity;
    size_t num_tasks = CEIL_DIV(n, rows_per_task);

    for (size_t i = 0; i < num_tasks; i++ ){
        size_t start_row = i * rows_per_task;
        size_t end_row   = MIN(start_row + rows_per_task, n);
        size_t num_rows = end_row - start_row;

        float *a_data = (float*) &a->data[start_row * p];
        float *b_data = (float*) b->data;
        float *c_data = &c->data[start_row * m];

        if (device == DEVICE_CPU) {
            task_submatrix_multiply_cpu(
                    num_rows, m, p,
                    a_data, b_data, c_data);
        }

        else if (device == DEVICE_GPU) {
            task_submatrix_multiply_gpu(
                    num_rows, m, p,
                    a_data, b_data, c_data);
        }

        else {
            task_submatrix_multiply_hybrid(
                    num_rows, m, p,
                    a_data, b_data, c_data);
        }
    }

    // force synchronization
#pragma omp taskwait

}


int main(int argc, char *argv[]) {
    benchmark_platform(argc, argv,
            "OmpSs",
            NULL,
            compute,
            NULL);
}
