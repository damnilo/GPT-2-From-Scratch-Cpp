#ifndef GPT_2_FROM_SCRATCH_DEVICEGEMM_H
#define GPT_2_FROM_SCRATCH_DEVICEGEMM_H

#include "CudaGemm.h"
#include "CustomGemm.h"

#include <cstddef>

// Which device launch to use. The pointers are already on the device.
enum class GemmDevicePath {
    Cublas,
    Custom
};

inline bool gemmDevice(GemmDevicePath path, const float* a, const float* b, float* c,
    size_t rows, size_t common, size_t cols
) {
    if (path == GemmDevicePath::Custom) {
        return customGemmDevice(a, b, c, rows, common, cols);
    }
    return cudaGemmDevice(a, b, c, rows, common, cols);
}

inline bool gemmBatchDevice(GemmDevicePath path, const float* a, const float* b, float* c, 
    size_t batches, size_t rows, size_t commons, size_t cols
) {
    if (path == GemmDevicePath::Custom) {
        return customGemmBatchDevice(a, b, c, batches, rows, commons, cols);
    }    

    return cudaGemmBatchDevice(a, b, c, batches, rows, commons, cols);
}

#endif
