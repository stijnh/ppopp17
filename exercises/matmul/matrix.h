#pragma once

#include <stddef.h> // for size_t

#include "common.h"

typedef struct {
    size_t cols;
    size_t rows;
    float *data;
} matrix_t;


EXTERN_C matrix_t matrix_allocate(size_t rows, size_t cols);
EXTERN_C void matrix_free(matrix_t *m);

EXTERN_C matrix_t matrix_create(size_t rows, size_t cols, float *ptr);
EXTERN_C matrix_t matrix_extract_rows(matrix_t *m, size_t start_row, size_t end_row);
EXTERN_C void matrix_fill_random(matrix_t *m);
EXTERN_C void matrix_multiply(const matrix_t *a, const matrix_t *b, matrix_t *c);
