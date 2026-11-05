#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000
#define BLOCK_SIZE 256

__global__ void cudaVectorScaling(const float* in, float scale, float* out, int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < embedding_dim) {
        out[i] = in[i] * scale;
    }
}