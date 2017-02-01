#pragma once

#include <sys/time.h>
#include <time.h>
#include <ctype.h>

#define MIN(a, b) \
    ((a) < (b) ? (a) : (b))
#define MAX(a, b) \
    ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a, b) \
    (((a) / (b)) + ((a) % (b) != 0 ? 1 : 0))

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif


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

static int stricmp(const char *str1, const char *str2) {
    int v;
    char c1, c2;

    do {
        c1 = *str1++;
        c2 = *str2++;

        v = tolower(c1) - tolower(c2);
    } while ((v == 0) && c1 != 0 && c2 != 0);

    return v;
}
