#include "cuda_kernels.h"

#define BLOCK_SIZE (128)
#define MAX_GRID_SIZE (65535)

#define KERNEL_FOR_EACH(VAR, SIZE) \
    for (int VAR = threadIdx.x + blockDim.x * blockIdx.x; \
            VAR < SIZE; VAR += blockDim.x * gridDim.x)


static void get_launch_params(int size, dim3 *grid_size, dim3 *block_size) {
    *block_size = dim3(BLOCK_SIZE);
    *grid_size = dim3(MIN(CEIL_DIV(size, BLOCK_SIZE), MAX_GRID_SIZE));
}

static __global__ void kernel_horz_flip(
        int width,
        int height,
        unsigned char *data) {

    KERNEL_FOR_EACH(gid, (width / 2) * height) {
        int x = gid % (width / 2);
        int y = gid / (width / 2);
        int rx = width - x - 1;

        unsigned char tmp = data[y * width + rx];
        data[y * width + rx] = data[y * width + x];
        data[y * width + x] = tmp;
    }
}

void cuda_launch_horz_flip(
        cudaStream_t stream,
        int width,
        int height,
        unsigned char *dev_data) {
    dim3 grid_size, block_size;
    get_launch_params(width * height, &grid_size, &block_size);

    kernel_horz_flip<<<grid_size, block_size, 0, stream>>>(
            width, height, dev_data);
    CUDA_CHECK(cudaGetLastError);
}


static __global__ void kernel_find_min_max(
        int width,
        int height,
        const unsigned char *data,
        unsigned int *min,
        unsigned int *max) {

    __shared__ unsigned int shared_min;
    __shared__ unsigned int shared_max;
    int lid = threadIdx.x;

    if (lid == 0) {
        shared_min = 0xffffffff;
        shared_max = 0;
    }

    __syncthreads();

    KERNEL_FOR_EACH(gid, width * height) {
        unsigned int val = data[gid];

        if (val < shared_min)
            atomicMin(&shared_min, val);

        if (val > shared_max)
            atomicMax(&shared_max, val);
    }

    __syncthreads();

    if (lid == 0) {
        atomicMin(min, shared_min);
        atomicMax(max, shared_max);
    }
}


void cuda_launch_find_min_max(
        cudaStream_t stream,
        int width,
        int height,
        const unsigned char *dev_data,
        int *dev_min,
        int *dev_max) {
    CUDA_CHECK(cudaMemsetAsync,
            dev_min,
            0xff,
            sizeof(int),
            stream);

    CUDA_CHECK(cudaMemsetAsync,
            dev_max,
            0x00,
            sizeof(int),
            stream);

    dim3 grid_size, block_size;
    get_launch_params(width * height, &grid_size, &block_size);

    kernel_find_min_max<<<grid_size, block_size, 0, stream>>>(
            width, height, dev_data,
            (unsigned int*) dev_min, (unsigned int*) dev_max);
    CUDA_CHECK(cudaGetLastError);
}

static __global__ void kernel_smooth(
        int width,
        int height,
        const unsigned char *input,
        unsigned char *output) {

    KERNEL_FOR_EACH(gid, width * height) {
        int x = gid % width;
        int y = gid / width;
        int total = 0;

        for (int dx = x - 1; dx <= x + 1; dx++) {
            for (int dy = y - 1; dy <= y + 1; dy++) {
                if (dx >= 0 && dy >= 0 && dx < width && dy < height) {
                    total += input[dy * width + dx];
                }
            }
        }

        output[y * width + x] = (unsigned char)(total / 9);
    }
}

void cuda_launch_smooth(
        cudaStream_t stream,
        int width,
        int height,
        const unsigned char *dev_input,
        unsigned char *dev_output) {

    dim3 grid_size, block_size;
    get_launch_params(width * height, &grid_size, &block_size);

    kernel_smooth<<<grid_size, block_size, 0, stream>>>(
            width, height, dev_input, dev_output);
}

static __global__ void kernel_contrast(
        int width,
        int height,
        unsigned char *data,
        const int *__restrict__ min_ptr,
        const int *__restrict__ max_ptr) {

    int min = *min_ptr;
    int max = *max_ptr;

    double lower = 0.9 * min + 0.1 * max;
    double upper = 0.1 * min + 0.9 * max;

    KERNEL_FOR_EACH(gid, width * height) {
        int old_pixel = data[gid];
        int new_pixel;

        if (old_pixel <= lower) {
            new_pixel = 0;
        } else if (old_pixel >= upper) {
            new_pixel = 255;
        } else {
            new_pixel = (int)(255.0 * (old_pixel - lower) / (upper - lower));
        }

            
            
        data[gid] = (unsigned char)new_pixel;
    }
}

void cuda_launch_contrast(
        cudaStream_t stream,
        int width,
        int height,
        unsigned char *dev_data,
        int *dev_min,
        int *dev_max) {

    dim3 grid_size, block_size;
    get_launch_params(width * height, &grid_size, &block_size);

    kernel_contrast<<<grid_size, block_size, 0, stream>>>(
            width, height, dev_data, dev_min, dev_max);
    CUDA_CHECK(cudaGetLastError);
}
