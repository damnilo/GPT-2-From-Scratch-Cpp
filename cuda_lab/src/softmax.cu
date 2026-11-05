#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void cudaSoftmax(float* out, int batches, int sequence_len, int embedding_dim, int axis) {

    if (axis == -1) axis = 2;

    int base, stride, reduce_len;

    if (axis == 2) {
        int batch = blockIdx.x;
        int row = blockDim.y * blockIdx.y + threadIdx.y;

        if (batch >= batches || row >= sequence_len) return;

        base = batch * sequence_len * embedding_dim + row * embedding_dim;
        stride = 1;
        reduce_len = embedding_dim;
    } else if (axis == 1) {
        
        int batch = blockIdx.x;
        int col = blockDim.y * blockIdx.y + threadIdx.y;

        if (batch >= batches || col >= embedding_dim) return;

        base = batch * sequence_len * embedding_dim + col;
        stride = sequence_len;
        reduce_len = sequence_len;
    } else if (axis == 0) {
        
        int row = blockIdx.x;
        int col = blockDim.y * blockIdx.y + threadIdx.y;

        if (row >= sequence_len || col >= embedding_dim) return;

        base = row * embedding_dim + col;
        stride = sequence_len * embedding_dim;
        reduce_len = batches;
    }
    
    float max = -INFINITY;
    for (int l = 0; l < reduce_len; ++l){
        if (max < out[base + l * stride]) {
            max = out[base + l * stride];
        }
    }

    float sum = 0.0f;
    for (int l = 0; l < reduce_len; ++l) {
        sum += (out[base + l * stride] == -INFINITY) ? 0.0 : expf(out[base + l * stride] - max);
    }

    for (int l = 0; l < reduce_len; ++l) {
        if (out[base + l * stride] == -INFINITY) out[base + l * stride] = 0.0f;
        else out[base + l * stride] = expf(out[base + l * stride] - max) / sum;
    }
}