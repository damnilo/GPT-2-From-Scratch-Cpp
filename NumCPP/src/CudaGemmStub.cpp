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
