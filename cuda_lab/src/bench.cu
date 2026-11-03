#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "../include/cuda_check.h"
#include "tiled_gemm.cu"
#include "naive_gemm.cu"
#include "scale.cu"
#include "vector_add.cu"

#define N 10000000
#define BLOCK_SIZE 256


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

void cpuVectorScaling(const float* a, float scale, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] * scale;
    }
}

void cpuVectorAdd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void init_matrix(float *vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void init_vector(float* vec, int n) {
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
    float* h_c_tile = (float*)malloc(c_bytes);

    if (h_a == NULL || h_b == NULL || h_c_cpu == NULL || h_c_gpu == NULL || h_c_tile == NULL) {
    fprintf(stderr, "malloc nije uspeo");
    return 1;
    }

    srand(1);
    init_matrix(h_a, a_n);
    init_matrix(h_b, b_n);

    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;

    float* t_a = NULL;
    float* t_b = NULL;
    float* t_c = NULL;

    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));

    CUDA_CHECK(cudaMalloc(&t_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&t_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&t_c, c_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(t_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    for (int i = 0; i < 3; ++i) {
        cpuNaiveGemm(h_a, h_b, h_c_cpu, rows, cols, commons);
        cudaNaiveGemm<<<grid, block>>>(d_a, d_b, d_c, rows, cols, commons);
        cudaTiledGemm<<<grid, block>>>(t_a, t_b, t_c, rows, cols, commons);
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

    double gpu_tile_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaTiledGemm<<<grid, block>>>(t_a, t_b, t_c, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_tile_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double gpu_tile_avg = gpu_tile_time / 20.0;

    printf("CPU: %f ms\n", cpu_avg);
    printf("GPU: %f ms\n", gpu_avg);
    printf("GPU tile: %f ms\n", gpu_tile_avg);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_tile, t_c, c_bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    int mismatches_tile = 0;
    for (int i = 0; i < c_n; ++i) {
        if(fabsf(h_c_cpu[i] - h_c_gpu[i]) > atol) {
            if (mismatches == 0) {
                printf("prvi pad na indeksu %d: cpu %f gpu %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            }

            ++mismatches;
        }

        if (fabsf(h_c_cpu[i] - h_c_tile[i]) > atol) {
            if (mismatches_tile == 0) {
                   printf("prvi pad tile-a na indeksu %d: cpu %f tile gpu %f\n", i, h_c_cpu[i], h_c_tile[i]);
            }

            ++mismatches_tile;
        }
    } 

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    free(h_c_tile);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(t_a));
    CUDA_CHECK(cudaFree(t_b));
    CUDA_CHECK(cudaFree(t_c));

    if (mismatches != 0) {
        printf("naive gemm: %d razlika\n", mismatches);
        return 1;
    }

    if (mismatches_tile != 0) {
        printf("tiled gemm: %d razlika \n", mismatches_tile);
        return 1;
    }

    printf("naive and tile gemm: ok\n");
    return 0;
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

int run_add() {
    const size_t bytes = N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c_cpu = (float*)malloc(bytes);
    float* h_c_gpu = (float*)malloc(bytes);
    if (h_a == NULL || h_b == NULL || h_c_cpu == NULL || h_c_gpu == NULL) {
        fprintf(stderr, "malloc nije uspeo\n");
        return 1;
    }

    srand(1);
    init_vector(h_a, N);
    init_vector(h_b, N);

    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < 3; ++i) {
        cpuVectorAdd(h_a, h_b, h_c_cpu, N);
        cudaVectorAdd<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double cpu_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cpuVectorAdd(h_a, h_b, h_c_cpu, N);
        const auto end = std::chrono::steady_clock::now();
        cpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double cpu_avg = cpu_time / 20.0;

    double gpu_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaVectorAdd<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double gpu_avg = gpu_time / 20.0;

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
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (mismatches != 0) {
        printf("vector: %d razlika\n", mismatches);
        return 1;
    }

    printf("vector: ok\n");
    return 0;
}

int main() {

    if (run_add() != 0) {
        printf("Run add failed.");
        return 1;
    }

    if (run_scale(1.0f / sqrtf(8.0f)) != 0 || run_scale(1.0f / sqrtf(32.0f)) != 0) {
        printf("Run scale failed.");
        return 1;
    }

    if (run_case(2, 3, 2, 1e-3) != 0
        || run_case(64, 8, 64, 1e-3) != 0
        || run_case(64, 32, 128, 1e-3) != 0
        || run_case(64, 32, 5256, 0.01) != 0
        || run_case(7, 3, 5, 1e-3) != 0) {
        printf("Run case failed.");
        return 1;
    }

    return 0;
}