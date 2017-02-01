#include "framework.h"

void compute(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    size_t n = c->rows; // == a->rows
    size_t m = c->cols; // == b->cols
    size_t p = a->cols; // == b->rows

    const float *__restrict__ a_data = a->data;
    const float *__restrict__ b_data = b->data;
    float *__restrict__ c_data = c->data;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            float val = 0;

            for (size_t k = 0; k < p; k++) {
                val += a_data[i * p + k] * b_data[k * m + j];
            }

            c_data[i * m + j] = val;
        }
    }
}


int main(int argc, char *argv[]) {
    return benchmark_platform(argc, argv,
            "sequential reference",
            NULL,
            compute,
            NULL);
}
