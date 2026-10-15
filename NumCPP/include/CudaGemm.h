//
// Created by HP on 10/11/2026.
//

#ifndef GPT_2_FROM_SCRATCH_CUDAGEMM_H
#define GPT_2_FROM_SCRATCH_CUDAGEMM_H

#include <cstddef>

// Row-major C[rows, cols] = A[rows, common] @ B[common, cols].
// Returns false when CUDA is not part of this build, no GPU is visible, or the launch fails.
bool cudaGemm(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols);

// Row-major batches: A[batch, rows, common] @ B[batch, common, cols].
bool cudaBatchGemm(const float* a, const float* b, float* c, size_t batch, size_t rows, size_t common, size_t cols);

bool cudaGemmAvailable();
bool cudaDeviceLabel(char* buffer, size_t buffer_size);

#endif //GPT_2_FROM_SCRATCH_CUDAGEMM_H
