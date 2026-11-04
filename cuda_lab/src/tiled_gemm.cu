#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_DIM dim3(16, 16)
#define TILE 16

__global__ void cudaTiledGemm(const float *a, const float *b, float *c,
                            int rows, int cols, int commons) {
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float A[TILE][TILE];
    __shared__ float B[TILE][TILE];

    float sum = 0.0f;
    int phase_num = (commons + TILE - 1) / TILE; 

    for (int i = 0; i < phase_num; i++) {
        int k = i * TILE;

        if (row < rows && (k + tx) < commons) {
            A[ty][tx] = a[row * commons + (k + tx)];
        } else {
            A[ty][tx] = 0;
        }

        if (col < cols && (k + ty) < commons) {
            B[ty][tx] = b[(k + ty) * cols + col];
        } else {
            B[ty][tx] = 0;
        }

        __syncthreads();

        for (int j = 0; j < TILE; j++) {
            sum += A[ty][j] * B[j][tx];
        }

        __syncthreads();
    }

    if (row < rows && col < cols) {
        c[row * cols + col] = sum;
    }
}