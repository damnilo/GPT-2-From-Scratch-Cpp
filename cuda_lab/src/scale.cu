#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "../include/cuda_check.h"

#define N 10000000
#define BLOCK_SIZE 256

__global__ void cudaVectorScaling(const float* a, float scale, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * scale;
    }
}

void cpuVectorScaling(const float* a, float scale, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] * scale;
    }
}

void init_vector(float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int run_scale(float scale) {
    const size_t bytes = N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_c_cpu = (float*)malloc(bytes);
    float* h_c_gpu = (float*)malloc(bytes);
    if (h_a == NULL || h_c_cpu == NULL || h_c_gpu == NULL) {
        fprintf(stderr, "malloc nije uspeo\n");
        return 1;
    }

    srand(1);
    init_vector(h_a, N);

    float* d_a = NULL;
    float* d_c = NULL;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < 3; ++i) {
        cpuVectorScaling(h_a, scale, h_c_cpu, N);
        cudaVectorScaling<<<num_blocks, BLOCK_SIZE>>>(d_a, scale, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double cpu_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cpuVectorScaling(h_a, scale, h_c_cpu, N);
        const auto end = std::chrono::steady_clock::now();
        cpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double cpu_avg = cpu_time / 20.0;

    double gpu_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaVectorScaling<<<num_blocks, BLOCK_SIZE>>>(d_a, scale, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double gpu_avg = gpu_time / 20.0;

    printf("scale %f\n", scale);
    printf("CPU: %f ms\n", cpu_avg);
    printf("GPU: %f ms\n", gpu_avg);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            if (mismatches == 0) {
                printf("prvi pad na indeksu %d: cpu %f gpu %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            }
            ++mismatches;
        }
    }

    free(h_a);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_c));

    if (mismatches != 0) {
        printf("scale: %d razlika\n", mismatches);
        return 1;
    }

    printf("scale: ok\n");
    return 0;
}

int main() {
    if (run_scale(1.0f / sqrtf(8.0f)) != 0) {
        return 1;
    }
    return run_scale(1.0f / sqrtf(32.0f));
}
