#ifndef GPT_2_FROM_SCRATCH_WMMAGEMM_H
#define GPT_2_FROM_SCRATCH_WMMAGEMM_H

#include <cstddef>

// Row-major C[rows, cols] = A[rows, common] @ B[common, cols].
// A, B, and C are FP32 and already on the device. A and B are rounded to FP16.
// The product uses tensor cores and accumulates in FP32. rows, common, and cols
// must be multiples of 16. This path is separate from the FP32 GEMM.
bool wmmaGemmDevice(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols);

// out16 is a device buffer of `count` FP16 values.
bool wmmaQuantizeDevice(const float* in, void* out16, size_t count);

// a16 and b16 are device FP16 buffers filled by wmmaQuantizeDevice. c is FP32.
// This launch does not allocate or round.
bool wmmaGemmLaunch(const void* a16, const void* b16, float* c, size_t rows, size_t common, size_t cols);

#endif
