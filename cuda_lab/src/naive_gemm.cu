#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "../include/cuda_check.h"

#define BLOCK_DIM dim3(16, 16)

__global__ void cudaNaiveGemm(const float *a, const float *b, float *c,
                            int rows, int cols, int commons) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < cols && row < rows) { 
        float sum = 0.0f;

        for (int k = 0; k < commons; ++k) {
            sum += a[row * commons + k] * b[k * cols + col];
        }

        c[row * cols + col] = sum;
    }
}