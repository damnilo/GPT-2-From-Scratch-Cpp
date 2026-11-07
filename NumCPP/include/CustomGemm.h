#ifndef GPT_2_FROM_SCRATCH_CUSTOMGEMM_H
#define GPT_2_FROM_SCRATCH_CUSTOMGEMM_H

#include <cstddef>

bool customGemm(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols);

bool customBatchGemm(const float* a, const float* b, float* c, size_t batch, size_t rows, size_t common, size_t cols);

// A, B, and C are already on the device. This launch does not copy.
bool customGemmDevice(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols);

bool customGemmBatchDevice(const float* a, const float* b, float* c, size_t batches, size_t rows, size_t commons, size_t cols);

#endif
