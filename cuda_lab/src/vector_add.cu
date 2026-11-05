#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000
#define BLOCK_SIZE 256

__global__ void cudaVectorAdd(const float* in_a, const float* in_b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = in_a[i] + in_b[i];
    }
}