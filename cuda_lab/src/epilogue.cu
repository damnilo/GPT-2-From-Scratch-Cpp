#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void cudaEpilogueForward(float* in, int batches, int sequence_len, float scale) {

    int i = blockIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int base = i * sequence_len * sequence_len + j * sequence_len;
    float max = -INFINITY;

    if (i >= batches || j >= sequence_len) return;

    for (int k = 0; k < sequence_len; ++k) {
        in[base + k] = (j >= k) ? in[base + k] * scale : -INFINITY;

        max = (in[base + k] > max) ? in[base + k] : max;
    }

    float sum = 0.0f;
    for (int k = 0; k < sequence_len; ++k) {
        sum += (in[base + k] == -INFINITY) ? 0.0f : expf(in[base + k] - max);
    }

    for (int k = 0; k < sequence_len; ++k) {
        in[base + k] = (in[base + k] == -INFINITY) ? 0.0f : expf(in[base + k] - max) / sum;
    }
}

__global__ void cudaEpilogueBackward(float* grad_out, float* out, float* ret, int batches, int sequence_len, float scale) {
    
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= batches || j >= sequence_len) return;

    int base = i * sequence_len * sequence_len + j * sequence_len;

    for (int k = 0; k < sequence_len; ++k) {
        ret[base + k] = grad_out[base + k] * out[base + k];
    }

    float sum = 0.0f;
    for (int k = 0; k < sequence_len; ++k) {
        sum += ret[base + k];
    }

    for (int k = 0; k < sequence_len; ++k) {
        ret[base + k] = out[base + k] * (grad_out[base + k] - sum) * scale;
    }
}