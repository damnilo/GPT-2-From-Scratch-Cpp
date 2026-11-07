//
// Created by HP on 10/11/2026.
//

#include "../include/CudaGemm.h"
#include "../include/CustomGemm.h"

bool cudaGemm(const float*, const float*, float*, size_t, size_t, size_t) {
    return false;
}

bool cudaBatchGemm(const float*, const float*, float*, size_t, size_t, size_t, size_t) {
    return false;
}

bool customGemm(const float*, const float*, float*, size_t, size_t, size_t) {
    return false;
}

bool customBatchGemm(const float*, const float*, float*, size_t, size_t, size_t, size_t) {
    return false;
}

bool cudaGemmAvailable() {
    return false;
}

bool cudaDeviceLabel(char*, size_t) {
    return false;
}

bool cudaGemmDevice(const float*, const float*, float*, size_t, size_t, size_t) {
    return false;
}

bool customGemmDevice(const float*, const float*, float*, size_t, size_t, size_t) {
    return false;
}

bool cudaGemmBatchDevice(const float* a, const float* b, float* c, size_t batches, size_t rows, size_t commons, size_t cols) {
    return false;
}

bool customGemmBatchDevice(const float* a, const float* b, float* c, size_t batches, size_t rows, size_t commons, size_t cols) {
    return false;
}
