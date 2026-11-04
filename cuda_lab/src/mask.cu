#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

__global__ void cudaMask(float* a, int batches, int sequence_len) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < batches && k < sequence_len && j < k) {
        a[i * sequence_len * sequence_len + j * sequence_len + k] = -INFINITY;
    }
}