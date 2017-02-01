#pragma once

#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>

#define MIN(a, b) \
    ((a) < (b) ? (a) : (b))
#define MAX(a, b) \
    ((a) > (b) ? (a) : (b))
#define CEIL_DIV(a, b) \
    (((a) / (b)) + ((a) % (b) != 0 ? 1 : 0))

#define SWAP(x, y) \
    swap_impl((void*)&(x), (void*)&(y), sizeof(x));

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

inline static void swap_impl(void *lhs, void *rhs, int size) {
    char tmp[size];
    memcpy(tmp, lhs, size);
    memcpy(lhs, rhs, size);
    memcpy(rhs , tmp, size);
}

typedef struct timeval stopwatch_t;

static stopwatch_t timer() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time;
}

static double timer_diff(stopwatch_t before, stopwatch_t after) {
    return (after.tv_sec - before.tv_sec) +
        (after.tv_usec - before.tv_usec) / 1000000.0;

}

