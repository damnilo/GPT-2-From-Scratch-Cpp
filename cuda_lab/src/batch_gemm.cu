#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE 16
#define BATCH 3

__global__ void cudaNaiveBatchGemm(const float* a, const float* b, float* c,
                            int batches, int rows, int cols, int commons) {

    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.z * blockIdx.z + threadIdx.z;
    int batch = blockIdx.x;
    
    if (col < cols && row < rows && batch < batches) {
        float sum = 0.0f;

        for (int common = 0; common < commons; ++common) {
            sum += a[batch * rows * commons + row * commons + common] * b[batch * commons * cols + common * cols + col];
        }

        c[batch * rows * cols + row * cols + col] = sum;
    }
}

__global__ void cudaTiledBatchGemm(const float* a, const float* b, float* c,
                                int batches, int rows, int cols, int commons) {
    
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int col = blockIdx.y * TILE + ty;
    int row = blockIdx.z * TILE + tz;
    int batch = blockIdx.x;

    if (batch >= batches) {
        return;
    }

    const float* a_batch = a + batch * rows * commons;
    const float* b_batch = b + batch * commons * cols;
    float* c_batch = c + batch * rows * cols;

    __shared__ float A[TILE][TILE];
    __shared__ float B[TILE][TILE];

    float sum = 0.0f;
    int phase_num = (commons + TILE - 1) / TILE;

    for (int i = 0; i < phase_num; ++i) {
        int k = i * TILE;

        if (row < rows && (k + ty) < commons) {
            A[tz][ty] = a_batch[row * commons + (k + ty)];
        } else {
            A[tz][ty] = 0.0f;
        }

        if (col < cols && (k + tz) < commons) {
            B[tz][ty] = b_batch[(k + tz) * cols + col];
        } else {
            B[tz][ty] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE; ++j) {
            sum += A[tz][j] * B[j][ty];
        }

        __syncthreads();
    }

    if (row < rows && col < cols) {
        c_batch[row * cols + col] = sum;
    }

}