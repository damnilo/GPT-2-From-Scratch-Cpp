#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void cudaSoftmax(float* a, int batches, int sequence_len, int axis) {

    if (axis == -1) axis = 2;

    int base, stride, reduce_len;

    if (axis == 2) {
        int batch = blockIdx.x;
        int row = blockDim.y * blockIdx.y + threadIdx.y;

        if (batch >= batches || row >= sequence_len) return;

        base = batch * sequence_len * sequence_len + row * sequence_len;
        stride = 1;
        reduce_len = sequence_len;
    } else if (axis == 1) {
        
        int batch = blockIdx.x;
        int col = blockDim.y * blockIdx.y + threadIdx.y;

        if (batch >= batches || col >= sequence_len) return;

        base = batch * sequence_len * sequence_len + col;
        stride = sequence_len;
        reduce_len = sequence_len;
    } else if (axis == 0) {
        
        int row = blockIdx.x;
        int col = blockDim.y * blockIdx.y + threadIdx.y;

        if (row >= sequence_len || col >= sequence_len) return;

        base = row * sequence_len + col;
        stride = sequence_len * sequence_len;
        reduce_len = batches;
    }
    
    float max = -INFINITY;
    for (int l = 0; l < reduce_len; ++l){
        if (max < a[base + l * stride]) {
            max = a[base + l * stride];
        }
    }

    float sum = 0.0f;
    for (int l = 0; l < reduce_len; ++l) {
        sum += (a[base + l * stride] == -INFINITY) ? 0.0 : expf(a[base + l * stride] - max);
    }

    for (int l = 0; l < reduce_len; ++l) {
        if (a[base + l * stride] == -INFINITY) a[base + l * stride] = 0.0f;
        else a[base + l * stride] = expf(a[base + l * stride] - max) / sum;
    }
}