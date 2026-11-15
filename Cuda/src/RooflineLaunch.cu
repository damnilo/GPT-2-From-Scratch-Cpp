#include "../include/RooflineLaunch.h"
#include "../include/Kernels.cuh"

#include <cuda_runtime.h>

bool rooflineSoftmax(float* scores, size_t H, size_t T, float scale) {
    if (scores == nullptr || H == 0 || T == 0) {
        return false;
    }
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    cudaMaskSoftmax3d<<<grid, 1>>>(scores, H, T, scale);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

bool rooflineLayerNorm(float* x, float* y, float* gamma, float* beta, size_t T, size_t C) {
    if (x == nullptr || y == nullptr || gamma == nullptr || beta == nullptr || T == 0 || C == 0 || C > 1024) {
        return false;
    }
    const GpuTensor g{gamma, C};
    const GpuTensor b{beta, C};
    cudaLayernorm<<<static_cast<unsigned>(T), static_cast<unsigned>(C)>>>(x, y, g, b, T, C);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}
