#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "matrix.h"


matrix_t matrix_allocate(size_t rows, size_t cols) {
    matrix_t m;
    m.cols = cols;
    m.rows = rows;
    m.data = (float*) malloc(sizeof(float) * cols * rows);
    return m;
}


void matrix_free(matrix_t *m) {
    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}


void matrix_fill_random(matrix_t *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = rand() % 10;
    }
}


matrix_t matrix_create(size_t rows, size_t cols, float *ptr) {
    matrix_t m;
    m.rows = rows;
    m.cols = cols;
    m.data = ptr;
    return m;
}


matrix_t matrix_extract_rows(matrix_t *m, size_t start_row, size_t end_row) {
    if (start_row > end_row || end_row > m->rows) {
        fprintf(stderr, "error: cannot extract submatrix from row %d to row %d "
                "for matrix of size (%d, %d)\n",
                (int)start_row, (int)end_row,
                (int)m->rows, (int)m->cols);
        exit(EXIT_FAILURE);
    }

    return matrix_create(
        end_row - start_row,
        m->cols,
        m->data + start_row * m->cols);
}


void matrix_multiply(const matrix_t *a, const matrix_t *b, matrix_t *c) {
    if (a->rows != c->rows || a->cols != b->rows || b->cols != c->cols) {
        fprintf(stderr, "error: cannot multiply matrix (%d, %d) "
                "x (%d, %d) = (%d, %d)\n",
                (int) a->rows, (int) a->cols,
                (int) b->rows, (int) b->cols,
                (int) c->rows, (int) c->cols);
        exit(EXIT_FAILURE);
    }

    size_t n = c->rows;
    size_t m = c->cols;
    size_t p = a->cols;

    const float *__restrict__ a_data = a->data;
    const float *__restrict__ b_data = b->data;
    float *__restrict__ c_data = c->data;

    for (size_t idx = 0; idx < n * m; idx++) {
        c_data[idx] = 0;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < p; k++) {
            float a_val = a_data[i * p + k];
            const float *b_ptr = &b_data[k * m];
            float *c_ptr = &c_data[i * m];
            size_t remaining = m;

            while (remaining--) {
                *(c_ptr++) += a_val * *(b_ptr++);
            }
        }
    }
}
