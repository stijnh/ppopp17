#include <HPL.h>

#include "framework.h"

using namespace HPL;


// forward declarations
void init(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c);
void compute(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c);


// matrix multiply kernel, written in HPL language.
void kernel_matrix_mul(
        Array<float, 1> a,
        Array<float, 1> b,
        Array<float, 1> c,
        Int n, Int m, Int p) {

    Size_t i = idy;
    Size_t j = idx;
    Size_t k;
    Float val = 0.0;

    for_ (k = 0, k < p, k++) {
        val += a[i * p + k] * b[k * m + j];
    }

    c[i * m + j] = val;
}


// partitioner calls kernel for subset of arguments
void partitioner(FRunner& fr, Range ranges[3],
        Array<float, 1> &a,
        Array<float, 1> &b,
        Array<float, 1> &c,
        Int &n, Int &m, Int &p) {

    // find rows for this kernel
    size_t row_start = ranges[0].origin;
    size_t row_end = ranges[0].end + 1;

    // find offsets in matrix A
    size_t a_start = row_start * p.value();
    size_t a_end =   row_end   * p.value() - 1;

    // find offsets in matrix C
    size_t c_start = row_start * m.value();
    size_t c_end =   row_end   * m.value() - 1;

    fr(a(Range(a_start, a_end)), b, c(Range(c_start, c_end)), n, m, p);
}


// execution plan used for weights of partitioning, possible options for balancing:
// - NOSEARCH: user-defined weights
// - EXHAUS: exhaustive search by trying all weights.
// - ADAPT: adapt weights after each run based on measurements.
// - ADAPT2: similar to ADAPT, adjust until time difference falls below threshold.
static ExecutionPlan execution_plan(partitioner, HPL::NOSEARCH);


void init(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    int device = params->device;
    double alpha = params->alpha;
    const int total_weight = 10000;

    if (device == DEVICE_UNDEFINED) {
        device = DEVICE_HYBRID;
    }

    if (device & DEVICE_CPU) {
        execution_plan.add(Device(CPU), int(total_weight * alpha));
    }

    if (device & DEVICE_GPU) {
        execution_plan.add(Device(GPU), total_weight - int(total_weight * alpha));
    }

    // runs once to tune execution plan for hybrid platform
    if (device == DEVICE_HYBRID) {
        compute(params, a, b, c);
    }
}


void compute(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    int n = c->rows;
    int m = c->cols;
    int p = a->cols;

    Array<float, 1> array_a(a->cols * a->rows, a->data);
    Array<float, 1> array_b(b->cols * b->rows, b->data);
    Array<float, 1> array_c(c->cols * c->rows, c->data);

    Int n_arg = n;
    Int m_arg = m;
    Int p_arg = p;

    // execute kernel with given execution plan
    eval(kernel_matrix_mul)
        .global(n, m)
        .executionPlan(execution_plan)
        (array_a, array_b, array_c, n_arg, m_arg, p_arg);


    // force synchronization by request read access to matrix C
    array_c.data(HPL_RD);
}


void deinit(const params_t *params, const matrix_t *a, const matrix_t *b, matrix_t *c) {
    auto weights = execution_plan.getWeights();

    printf("final weights of execution plan:\n");
    printf(" - CPU: %d%%\n", (int) weights[0]);
    printf(" - GPU: %d%%\n", (int) weights[1]);
}



int main(int argc, char *argv[]) {
    return benchmark_platform(argc, argv,
            "HPL",
            init,
            compute,
            NULL);
}
