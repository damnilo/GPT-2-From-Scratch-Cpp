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
#include "cublas_v2.cu"
#include "batch_gemm.cu"
#include "mask.cu"
#include "softmax.cu"
#include "epilogue.cu"
#include "transpose.cu"
#include "attention.cu"

#define N 10000000
#define BLOCK_SIZE 256

void cpuMaskSoftmax(const float* in, float* out, int batches, int sequence_len) {
    const int n = batches * sequence_len * sequence_len;
    for (int i = 0; i < n; ++i) {
        out[i] = in[i];
    }

    for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < sequence_len; ++row) {
            for (int col = row + 1; col < sequence_len; ++col) {
                out[(batch * sequence_len + row) * sequence_len + col] = -INFINITY;
            }
        }
    }

    for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < sequence_len; ++row) {
            const int base = (batch * sequence_len + row) * sequence_len;
            float max = -INFINITY;
            for (int col = 0; col < sequence_len; ++col) {
                if (max < out[base + col]) {
                    max = out[base + col];
                }
            }

            float sum = 0.0f;
            for (int col = 0; col < sequence_len; ++col) {
                const float x = out[base + col];
                sum += (x == -INFINITY) ? 0.0f : expf(x - max);
            }

            for (int col = 0; col < sequence_len; ++col) {
                const float x = out[base + col];
                out[base + col] = (x == -INFINITY) ? 0.0f : expf(x - max) / sum;
            }
        }
    }
}

void cpuTranspose(const float* in, float* out, int batches, int sequence_len, int embedding_dim) {
    for (int i = 0; i < batches; ++i) {
        for (int j = 0; j < sequence_len; ++j) {
            for (int k = 0; k < embedding_dim; ++k) {
                out[i * sequence_len * embedding_dim + k * sequence_len + j] = in[i * sequence_len * embedding_dim + j * embedding_dim + k];
            }
        }
    }
}

void cpuNaiveBatchGemm(const float *a, const float *b, float *c,
    int batches, int rows, int cols, int commons) {
    for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                float sum = 0.0f;

                for (int comm = 0; comm < commons; ++comm) {
                    sum += a[batch * rows * commons + row * commons + comm] * b[batch * commons * cols + comm * cols + col];
                }

                c[batch * rows * cols + row * cols + col] = sum;
            }
        }
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

int run_batch_gemm(int batches, int rows, int commons, int cols, float atol) {
    const size_t a_n = (size_t)batches * rows * commons;
    const size_t b_n = (size_t)batches * commons * cols;
    const size_t c_n = (size_t)batches * rows * cols;
    
    const size_t a_bytes = a_n * sizeof(float);
    const size_t b_bytes = b_n * sizeof(float);
    const size_t c_bytes = c_n * sizeof(float);

    float* h_a = (float*)malloc(a_bytes);
    float* h_b = (float*)malloc(b_bytes);
    float* h_c_cpu = (float*)malloc(c_bytes);
    float* h_c_gpu = (float*)malloc(c_bytes);
    float* h_c_tile = (float*)malloc(c_bytes);
    float* h_cublas = (float*)malloc(c_bytes);

    if (h_a == NULL
        || h_b == NULL
        || h_c_cpu == NULL
        || h_c_gpu == NULL
        || h_c_tile == NULL
        || h_cublas == NULL) {
        fprintf(stderr, "malloc nije uspeo, batchGemm");
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        free(h_c_tile);
        free(h_cublas);
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

    dim3 block(1, 16, 16);
    dim3 grid(batches, (cols + 15) / 16, (rows + 15) / 16);

    int status = 0;
    int mismatches = 0;
    int mismatches_tile = 0;
    int mismatches_cublas = 0;
    double cpu_time = 0.0;
    double cpu_avg = 0.0;
    double gpu_time = 0.0;
    double gpu_avg = 0.0;
    double gpu_tile_time = 0.0;
    double gpu_tile_avg = 0.0;
    double cublas_time = 0.0;
    double cublas_avg = 0.0;

    for (int i = 0; i < 3; ++i) {
        cpuNaiveBatchGemm(h_a, h_b, h_c_cpu, batches, rows, cols, commons);
        cudaNaiveBatchGemm<<<grid, block>>>(d_a, d_b, d_c, batches, rows, cols, commons);
        cudaTiledBatchGemm<<<grid, block>>>(t_a, t_b, t_c, batches, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cpuNaiveBatchGemm(h_a, h_b, h_c_cpu, batches, rows, cols, commons);
        const auto end = std::chrono::steady_clock::now();
        cpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    } 
    cpu_avg = cpu_time / 20.0;

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaNaiveBatchGemm<<<grid, block>>>(d_a, d_b, d_c, batches, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    } 
    gpu_avg = gpu_time / 20.0;
    
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaTiledBatchGemm<<<grid, block>>>(t_a, t_b, t_c, batches, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_tile_time += std::chrono::duration<double, std::milli>(end - start).count();
    } 
    gpu_tile_avg = gpu_tile_time / 20.0;

    if (!cublasBatchGemm(h_a, h_b, h_cublas, batches, rows, commons, cols)) {
        printf("cuBLAS batch gemm failed.");
        status = 1;
        goto cleanup;
    }

    for (int i = 0; i < 2; ++i) {
        if (!cublasBatchGemmOnly(batches, rows, commons, cols)) {
            status = 1;
            goto cleanup;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (!cublasBatchGemmOnly(batches, rows, commons, cols)) {
            status = 1;
            goto cleanup;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        cublas_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    cublas_avg = cublas_time / 20.0;

    printf("CPU batch: %f ms\n", cpu_avg);
    printf("GPU batch: %f ms\n", gpu_avg);
    printf("GPU tile batch: %f ms\n", gpu_tile_avg);
    printf("cuBLAS batch: %f ms\n", cublas_avg);
    
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_tile, t_c, c_bytes, cudaMemcpyDeviceToHost));

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

        if (fabsf(h_c_cpu[i] - h_cublas[i]) > atol) {
            if (mismatches_cublas == 0) {
                printf("prvi pad cublas na indeksu %d: cpu %f cublas %f\n", i, h_c_cpu[i], h_cublas[i]);
            }

            ++mismatches_cublas;
        }
    } 

    if (mismatches != 0) {
        printf("naive batch gemm: %d razlika\n", mismatches);
        status = 1;
    } else if (mismatches_tile != 0) {
        printf("tiled batch gemm: %d razlika\n", mismatches_tile);
        status = 1;
    } else if (mismatches_cublas != 0) {
        printf("cublas batch gemm: %d razlika\n", mismatches_cublas);
        status = 1;
    } else {
        printf("naive, tile and cublas batch gemm: ok\n");
    }

    cleanup:
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        free(h_c_tile);
        free(h_cublas);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(t_a));
        CUDA_CHECK(cudaFree(t_b));
        CUDA_CHECK(cudaFree(t_c));
        return status;

}

int run_gemm(int rows, int commons, int cols, float atol) {
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
    float* h_cublas = (float*)malloc(c_bytes);

    if (h_a == NULL 
        || h_b == NULL 
        || h_c_cpu == NULL 
        || h_c_gpu == NULL 
        || h_c_tile == NULL
        || h_cublas == NULL) {
        fprintf(stderr, "malloc nije uspeo");
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        free(h_c_tile);
        free(h_cublas);
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

    int status = 0;
    int mismatches = 0;
    int mismatches_tile = 0;
    int mismatches_cublas = 0;
    double cpu_time = 0.0;
    double cpu_avg = 0.0;
    double gpu_time = 0.0;
    double gpu_avg = 0.0;
    double gpu_tile_time = 0.0;
    double gpu_tile_avg = 0.0;
    double cublas_time = 0.0;
    double cublas_avg = 0.0;

    for (int i = 0; i < 3; ++i) {
        cpuNaiveGemm(h_a, h_b, h_c_cpu, rows, cols, commons);
        cudaNaiveGemm<<<grid, block>>>(d_a, d_b, d_c, rows, cols, commons);
        cudaTiledGemm<<<grid, block>>>(t_a, t_b, t_c, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cpuNaiveGemm(h_a, h_b, h_c_cpu, rows, cols, commons);
        const auto end = std::chrono::steady_clock::now();
        cpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    cpu_avg = cpu_time / 20.0;

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaNaiveGemm<<<grid, block>>>(d_a, d_b, d_c, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    gpu_avg = gpu_time / 20.0;

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cudaTiledGemm<<<grid, block>>>(t_a, t_b, t_c, rows, cols, commons);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        gpu_tile_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    gpu_tile_avg = gpu_tile_time / 20.0;

    if (!cublasGemm(h_a, h_b, h_cublas, rows, commons, cols)) {
        printf("cuBLAS gemm failed.");
        status = 1;
        goto cleanup;
    }

    for (int i = 0; i < 2; ++i) {
        if (!cublasGemmOnly(rows, commons, cols)) {
            status = 1;
            goto cleanup;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (!cublasGemmOnly(rows, commons, cols)) {
            status = 1;
            goto cleanup;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        cublas_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    cublas_avg = cublas_time / 20.0;

    printf("CPU: %f ms\n", cpu_avg);
    printf("GPU: %f ms\n", gpu_avg);
    printf("GPU tile: %f ms\n", gpu_tile_avg);
    printf("cuBLAS: %f ms\n", cublas_avg);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_tile, t_c, c_bytes, cudaMemcpyDeviceToHost));

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

        if (fabsf(h_c_cpu[i] - h_cublas[i]) > atol) {
            if (mismatches_cublas == 0) {
                printf("prvi pad cublas na indeksu %d: cpu %f cublas %f\n", i, h_c_cpu[i], h_cublas[i]);
            }

            ++mismatches_cublas;
        }
    } 

    if (mismatches != 0) {
        printf("naive gemm: %d razlika\n", mismatches);
        status = 1;
    } else if (mismatches_tile != 0) {
        printf("tiled gemm: %d razlika\n", mismatches_tile);
        status = 1;
    } else if (mismatches_cublas != 0) {
        printf("cublas gemm: %d razlika\n", mismatches_cublas);
        status = 1;
    } else {
        printf("naive, tile and cublas gemm: ok\n");
    }

    cleanup:
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        free(h_c_tile);
        free(h_cublas);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(t_a));
        CUDA_CHECK(cudaFree(t_b));
        CUDA_CHECK(cudaFree(t_c));
        return status;

}

int run_scale(float scale) {
    const size_t bytes = N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_c_cpu = (float*)malloc(bytes);
    float* h_c_gpu = (float*)malloc(bytes);
    if (h_a == NULL || h_c_cpu == NULL || h_c_gpu == NULL) {
        fprintf(stderr, "malloc nije uspeo\n");
        free(h_a);
        free(h_c_cpu);
        free(h_c_gpu);
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
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
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

int runScaleMaskSoftmax(int batches, int sequence_len, int embedding_dim) {
    const size_t a_n = (size_t)batches * sequence_len * sequence_len;
    const size_t bytes = a_n * sizeof(float);
    float scale = 1/sqrtf(embedding_dim);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_cpu = (float*)malloc(bytes);
    float* h_gpu = (float*)malloc(bytes);
    if (h_a == NULL || h_b == NULL || h_cpu == NULL || h_gpu == NULL) {
        fprintf(stderr, "malloc nije uspeo\n");
        free(h_a);
        free(h_b);
        free(h_cpu);
        free(h_gpu);
        return 1;
    }

    srand(1);
    init_matrix(h_a, (int)a_n);
    cpuVectorScaling(h_a, scale, h_b, (int)a_n);
    cpuMaskSoftmax(h_b, h_cpu, batches, sequence_len);

    float* d_a = NULL;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    dim3 block(1, 16, 1);
    dim3 grid(batches, (sequence_len + 15) / 16, 1);

    cudaScaleMaskSoftmax<<<grid, block>>>(d_a, batches, sequence_len, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu, d_a, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < (int)a_n; ++i) {
        if (fabsf(h_cpu[i] - h_gpu[i]) > 1e-5f) {
            if (mismatches == 0) {
                printf("prvi pad na indeksu %d: cpu %f gpu %f\n", i, h_cpu[i], h_gpu[i]);
            }
            ++mismatches;
        }
    }

    free(h_a);
    free(h_b);
    free(h_cpu);
    free(h_gpu);
    CUDA_CHECK(cudaFree(d_a));

    if (mismatches != 0) {
        printf("scale mask softmax batches=%d T=%d d=%d: %d razlika\n", batches, sequence_len, embedding_dim, mismatches);
        return 1;
    }

    printf("scale mask softmax batches=%d T=%d d=%d: ok\n", batches, sequence_len, embedding_dim);
    return 0;
}

int runMaskSoftmax(int batches, int sequence_len) {
    const size_t a_n = (size_t)batches * sequence_len * sequence_len;
    const size_t bytes = a_n * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_cpu = (float*)malloc(bytes);
    float* h_gpu = (float*)malloc(bytes);
    if (h_a == NULL || h_cpu == NULL || h_gpu == NULL) {
        fprintf(stderr, "malloc nije uspeo\n");
        free(h_a);
        free(h_cpu);
        free(h_gpu);
        return 1;
    }

    srand(1);
    init_matrix(h_a, (int)a_n);
    cpuMaskSoftmax(h_a, h_cpu, batches, sequence_len);

    float* d_a = NULL;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    dim3 mask_block(1, 16, 16);
    dim3 mask_grid(batches, (sequence_len + 15) / 16, (sequence_len + 15) / 16);

    dim3 block(1, 16, 1);
    dim3 grid(batches, (sequence_len + 15) / 16, 1);

    cudaMask<<<mask_grid, mask_block>>>(d_a, batches, sequence_len, sequence_len);
    CUDA_CHECK(cudaGetLastError());
    cudaSoftmax<<<grid, block>>>(d_a, batches, sequence_len, sequence_len, -1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu, d_a, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < (int)a_n; ++i) {
        if (fabsf(h_cpu[i] - h_gpu[i]) > 1e-5f) {
            if (mismatches == 0) {
                printf("prvi pad na indeksu %d: cpu %f gpu %f\n", i, h_cpu[i], h_gpu[i]);
            }
            ++mismatches;
        }
    }

    free(h_a);
    free(h_cpu);
    free(h_gpu);
    CUDA_CHECK(cudaFree(d_a));

    if (mismatches != 0) {
        printf("mask softmax batches=%d T=%d: %d razlika\n", batches, sequence_len, mismatches);
        return 1;
    }

    printf("mask softmax batches=%d T=%d: ok\n", batches, sequence_len);
    return 0;
}

int runAttention(int batches, int sequence_len, int embedding_dim) {
    size_t a_n = (size_t)batches * sequence_len * embedding_dim;
    size_t b_n = (size_t)batches * sequence_len * sequence_len;
    float scale = 1/sqrtf(embedding_dim);

    size_t a_bytes = a_n * sizeof(float);
    size_t b_bytes = b_n * sizeof(float);

    float* Q = (float*)malloc(a_bytes);
    float* K = (float*)malloc(a_bytes);
    float* V = (float*)malloc(a_bytes);
    float* K_t = (float*)malloc(a_bytes);
    float* out = (float*)malloc(a_bytes);
    float* gpu_out = (float*)malloc(a_bytes);
    float* scores = (float*)malloc(b_bytes);

    if (
        Q == NULL
        || K == NULL
        || V == NULL
        || K_t == NULL
        || out == NULL
        || gpu_out == NULL
        || scores == NULL
    ) {
        fprintf(stderr, "malloc nije uspeo\n");
        free(Q);
        free(K);
        free(V);
        free(K_t);
        free(out);
        free(gpu_out);
        free(scores);
        return 1;
    }

    srand(1);
    init_matrix(Q, (int)a_n);
    init_matrix(K, (int)a_n);
    init_matrix(V, (int)a_n);
    cpuTranspose(K, K_t, batches, sequence_len, embedding_dim);
    cpuNaiveBatchGemm(Q, K_t, scores, batches, sequence_len, sequence_len, embedding_dim);
    cpuVectorScaling(scores, scale, scores, (int)b_n);
    cpuMaskSoftmax(scores, scores, batches, sequence_len);
    cpuNaiveBatchGemm(scores, V, out, batches, sequence_len, embedding_dim, sequence_len);

    float* d_Q = NULL;
    float* d_K = NULL;
    float* d_V = NULL;
    float* d_K_t = NULL;
    float* d_out = NULL;
    float* d_scores = NULL;

    CUDA_CHECK(cudaMalloc(&d_Q, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_K_t, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, b_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, Q, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, a_bytes, cudaMemcpyHostToDevice));

    cudaAttentionForward(d_K, d_Q, d_V, d_K_t, d_out, d_scores, batches, sequence_len, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gpu_out, d_out, a_bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < (int)a_n; ++i) {
        if (fabsf(out[i] - gpu_out[i]) > 1e-5f) {
            if (mismatches == 0) {
                printf("prvi pad na indeksu %d: cpu %f gpu %f\n", i, out[i], gpu_out[i]);
            }
            ++mismatches;
        }
    }

    free(Q);
    free(K);
    free(V);
    free(K_t);
    free(out);
    free(gpu_out);
    free(scores);
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_K_t));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scores));

    if (mismatches != 0) {
        printf("attention batches=%d T=%d d=%d: %d razlika\n", batches, sequence_len, embedding_dim, mismatches);
        return 1;
    }

    printf("attention batches=%d T=%d d=%d: ok\n", batches, sequence_len, embedding_dim);
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

    if (run_gemm(2, 3, 2, 1e-3) != 0
        || run_gemm(64, 8, 64, 1e-3) != 0
        || run_gemm(64, 32, 128, 1e-3) != 0
        || run_gemm(64, 32, 5256, 0.01) != 0
        || run_gemm(7, 3, 5, 1e-3) != 0) {
        printf("Run gemm failed.");
        return 1;
    }

    if (run_batch_gemm(3, 2, 3, 2, 1e-3) != 0
        || run_batch_gemm(3, 64, 8, 64, 1e-3) != 0
        || run_batch_gemm(3, 64, 32, 128, 1e-3) != 0
        || run_batch_gemm(3, 64, 32, 5256, 0.01) != 0
        || run_batch_gemm(3, 7, 3, 5, 1e-3) != 0) {
        printf("Run batch gemm failed.");
        return 1;
    }

    if (runMaskSoftmax(1, 64) != 0
        || runMaskSoftmax(2, 64) != 0
        || runMaskSoftmax(1, 7) != 0) {
        printf("Run mask and softmax failed.");
        return 1;
    }

    if (runScaleMaskSoftmax(1, 64, 8) != 0
        || runScaleMaskSoftmax(2, 64, 8) != 0
        || runScaleMaskSoftmax(1, 7, 5) != 0) {
        printf("Run scale mask and softmax failed.");
        return 1;
    }

    if (runAttention(1, 64, 8) != 0
        || runAttention(1, 64, 32) != 0
        || runAttention(2, 64, 8) != 0) {
        printf("Run attention failed.");
        return 1;
    }

    return 0;
}