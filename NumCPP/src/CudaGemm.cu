//
// Created by HP on 10/11/2026.
//

#include "../include/CudaGemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdio>

namespace {

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t capacity = 0;

    bool ensure(size_t count) {
        if (count <= capacity) {
            return true;
        }

        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            capacity = 0;
        }

        if (cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)) != cudaSuccess) {
            return false;
        }

        capacity = count;
        return true;
    }
};

cublasHandle_t handle = nullptr;
DeviceBuffer buffer_a;
DeviceBuffer buffer_b;
DeviceBuffer buffer_c;
bool unavailable = false;
bool announced = false;

bool ready() {
    if (handle != nullptr) {
        return true;
    }

    if (unavailable) {
        return false;
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
        unavailable = true;
        return false;
    }

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        handle = nullptr;
        unavailable = true;
        return false;
    }

    cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);

    if (!announced) {
        announced = true;
        std::fputs("CUDA GEMM enabled.\n", stdout);
        std::fflush(stdout);
    }

    return true;
}

bool fitsInt(size_t rows, size_t common, size_t cols) {
    return rows <= static_cast<size_t>(INT_MAX) &&
           common <= static_cast<size_t>(INT_MAX) &&
           cols <= static_cast<size_t>(INT_MAX) &&
           rows > 0 && common > 0 && cols > 0;
}

}

bool cudaGemm(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols) {
    if (!fitsInt(rows, common, cols) || !ready()) {
        return false;
    }

    const size_t a_count = rows * common;
    const size_t b_count = common * cols;
    const size_t c_count = rows * cols;

    if (!buffer_a.ensure(a_count) || !buffer_b.ensure(b_count) || !buffer_c.ensure(c_count)) {
        return false;
    }

    if (cudaMemcpy(buffer_a.ptr, a, a_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(buffer_b.ptr, b, b_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int n = static_cast<int>(cols);
    const int m = static_cast<int>(rows);
    const int k = static_cast<int>(common);

    // Row-major A @ B is column-major B^T @ A^T, which is what this argument order computes.
    const cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        buffer_b.ptr,
        n,
        buffer_a.ptr,
        k,
        &beta,
        buffer_c.ptr,
        n
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    return cudaMemcpy(c, buffer_c.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool cudaBatchGemm(const float* a, const float* b, float* c, size_t batch, size_t rows, size_t common, size_t cols) {
    if (batch == 0 || batch > static_cast<size_t>(INT_MAX) || !fitsInt(rows, common, cols) || !ready()) {
        return false;
    }

    const size_t a_one = rows * common;
    const size_t b_one = common * cols;
    const size_t c_one = rows * cols;

    if (!buffer_a.ensure(batch * a_one) || !buffer_b.ensure(batch * b_one) || !buffer_c.ensure(batch * c_one)) {
        return false;
    }

    if (cudaMemcpy(buffer_a.ptr, a, batch * a_one * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(buffer_b.ptr, b, batch * b_one * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int n = static_cast<int>(cols);
    const int m = static_cast<int>(rows);
    const int k = static_cast<int>(common);

    const cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        buffer_b.ptr,
        n,
        static_cast<long long>(b_one),
        buffer_a.ptr,
        k,
        static_cast<long long>(a_one),
        &beta,
        buffer_c.ptr,
        n,
        static_cast<long long>(c_one),
        static_cast<int>(batch)
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    return cudaMemcpy(c, buffer_c.ptr, batch * c_one * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool cudaGemmAvailable() {
    return ready();
}

bool cudaDeviceLabel(char* buffer, size_t buffer_size) {
    if (buffer == nullptr || buffer_size == 0 || !ready()) {
        return false;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }

    std::snprintf(buffer, buffer_size, "%s", prop.name);
    return true;
}
