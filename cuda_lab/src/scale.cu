#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000
#define BLOCK_SIZE 256

__global__ void cudaVectorScaling(const float* a, float scale, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * scale;
    }
}