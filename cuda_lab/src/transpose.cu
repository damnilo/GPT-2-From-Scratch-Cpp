#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void cudaBatchTranspose(const float* in, float* out, int batches, int sequence_len, int embedding_dim) {
    
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= batches || j >= sequence_len || k >= embedding_dim) return;

    out[i * sequence_len * embedding_dim + k * sequence_len + j] = in[i * sequence_len * embedding_dim + j * embedding_dim + k];
}