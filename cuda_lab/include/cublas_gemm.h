
#ifndef GPT_2_FROM_SCRATCH_CUDAGEMM_H
#define GPT_2_FROM_SCRATCH_CUDAGEMM_H

#include <cstddef>

bool cublasGemm(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols);
bool cublasBatchGemm(const float* a, const float* b, float* c, size_t batch, size_t rows, size_t common, size_t cols);

bool cublasGemmAvailable();
bool cublasDeviceLabel(char* buffer, size_t buffer_size);

bool cublasGemmOnly(size_t rows, size_t common, size_t cols);
bool cublasBatchGemmOnly(size_t batches, size_t rows, size_t commons, size_t cols);

#endif 
