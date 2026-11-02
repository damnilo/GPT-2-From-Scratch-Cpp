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

void cpuNaiveGemm(const float *a, const float *b, float *c,
                int rows, int cols, int commons) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < commons; ++k) {
                sum += a[i * commons + k] * b [k * cols + j];
            }

            c[i * cols + j] = sum;
        }
    }
}

void init_matrix(float *vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int run_case(int rows, int commons, int cols, float atol) {
    const size_t a_n = (size_t)rows * commons;
    const size_t b_n = (size_t)commons * cols;
    const size_t c_n = (size_t)rows * cols;

    const size_t a_bytes = a_n * sizeof(float);
    const size_t b_bytes = b_n * sizeof(float);
    const size_t c_bytes = c_n * sizeof(float);

    float* h_a = (float*)malloc(a_bytes);
    float* h_b = (float*)malloc(b_bytes);
    float* h_c_cpu = (float*)malloc(c_bytes);
    float* h_c_gpu = (float*)malloc(c_bytes);

    if (h_a == NULL || h_b == NULL || h_c_cpu == NULL || h_c_gpu == NULL) {
        fprintf(stderr, "malloc nije uspeo");
        return 1;
    }

    srand(1);
    init_matrix(h_a, a_n);
    init_matrix(h_b, b_n);

    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;

    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    for (int i = 0; i < 3; ++i) {
        cpuNaiveGemm(h_a, h_b, h_c_cpu, rows, cols, commons);
        cudaNaiveGemm<<<grid, block>>>(d_a, d_b, d_c, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double cpu_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cpuNaiveGemm(h_a, h_b, h_c_cpu, rows, cols, commons);
        const auto end = std::chrono::steady_clock::now();
        cpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double cpu_avg = cpu_time / 20.0;

    double gpu_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaNaiveGemm<<<grid, block>>>(d_a, d_b, d_c, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double gpu_avg = gpu_time / 20.0;

    printf("CPU: %f ms\n", cpu_avg);
    printf("GPU: %f ms\n", gpu_avg);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, c_bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < c_n; ++i) {
        if(fabsf(h_c_cpu[i] - h_c_gpu[i]) > atol) {
            if (mismatches == 0) {
                printf("prvi pad na indeksu %d: cpu %f gpu %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            }

            ++mismatches;
        }
    } 

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (mismatches != 0) {
        printf("naive gemm: %d razlika\n", mismatches);
        return 1;
    }
    
    printf("naive gemm: ok\n");
    return 0;
}

int main() {

    if (run_case(2, 3, 2, 1e-3) != 0
        || run_case(64, 8, 64, 1e-3) != 0
        || run_case(64, 32, 128, 1e-3) != 0
        || run_case(64, 32, 5256, 0.01) != 0
        || run_case(7, 3, 5, 1e-3) != 0) {
        return 1;
    }

    return 0;
}