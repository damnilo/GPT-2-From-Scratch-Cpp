#include "../include/CustomGemm.h"

#include <cuda_runtime.h>

#include <climits>

#define TILE 16

__global__ void cudaTiledGemm(const float* a, const float* b, float* c,
    int rows, int commons, int cols) {

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float A[TILE][TILE];
    __shared__ float B[TILE][TILE];

    float sum = 0.0f;
    int phase_num = (commons + TILE - 1) / TILE;

    for (int i = 0; i < phase_num; i++) {
        int k = i * TILE;

        if (row < rows && (k + tx) < commons) {
            A[ty][tx] = a[row * commons + (k + tx)];
        } else {
            A[ty][tx] = 0.0f;
        }

        if (col < cols && (k + ty) < commons) {
            B[ty][tx] = b[(k + ty) * cols + col];
        } else {
            B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE; j++) {
            sum += A[ty][j] * B[j][tx];
        }

        __syncthreads();
    }

    if (row < rows && col < cols) {
        c[row * cols + col] = sum;
    }
}

__global__ void cudaTiledBatchGemm(const float* a, const float* b, float* c,
    int batches, int rows, int commons, int cols) {

    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int col = blockIdx.y * TILE + ty;
    int row = blockIdx.z * TILE + tz;
    int batch = blockIdx.x;

    if (batch >= batches) {
        return;
    }

    const float* a_batch = a + batch * rows * commons;
    const float* b_batch = b + batch * commons * cols;
    float* c_batch = c + batch * rows * cols;

    __shared__ float A[TILE][TILE];
    __shared__ float B[TILE][TILE];

    float sum = 0.0f;
    int phase_num = (commons + TILE - 1) / TILE;

    for (int i = 0; i < phase_num; ++i) {
        int k = i * TILE;

        if (row < rows && (k + ty) < commons) {
            A[tz][ty] = a_batch[row * commons + (k + ty)];
        } else {
            A[tz][ty] = 0.0f;
        }

        if (col < cols && (k + tz) < commons) {
            B[tz][ty] = b_batch[(k + tz) * cols + col];
        } else {
            B[tz][ty] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE; ++j) {
            sum += A[tz][j] * B[j][ty];
        }

        __syncthreads();
    }

    if (row < rows && col < cols) {
        c_batch[row * cols + col] = sum;
    }
}

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

DeviceBuffer buffer_a;
DeviceBuffer buffer_b;
DeviceBuffer buffer_c;
bool checked = false;
bool available = false;

bool ready() {
    if (checked) {
        return available;
    }

    checked = true;
    int devices = 0;
    available = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
    return available;
}

bool fitsInt(size_t rows, size_t common, size_t cols) {
    return rows <= static_cast<size_t>(INT_MAX) &&
           common <= static_cast<size_t>(INT_MAX) &&
           cols <= static_cast<size_t>(INT_MAX) &&
           rows > 0 && common > 0 && cols > 0;
}

bool launchOk() {
    return cudaGetLastError() == cudaSuccess;
}

}

bool customGemm(const float* a, const float* b, float* c,
    size_t rows, size_t common, size_t cols
) {
    if (!fitsInt(rows, common, cols) || !ready()) {
        return false;
    }

    const size_t a_count = rows * common;
    const size_t b_count = common * cols;
    const size_t c_count = rows * cols;

    if (!buffer_a.ensure(a_count)
        || !buffer_b.ensure(b_count)
        || !buffer_c.ensure(c_count)) {
        return false;
    }

    if (cudaMemcpy(buffer_a.ptr, a, a_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(buffer_b.ptr, b, b_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    const auto i = static_cast<int>(rows);
    const auto j = static_cast<int>(common);
    const auto k = static_cast<int>(cols);
    dim3 block(16, 16);
    dim3 grid((k + 15) / 16, (i + 15) / 16);

    cudaTiledGemm<<<grid, block>>>(
        buffer_a.ptr,
        buffer_b.ptr,
        buffer_c.ptr,
        i, j, k
    );
    if (!launchOk()) {
        return false;
    }

    return cudaMemcpy(c, buffer_c.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool customBatchGemm(const float* a, const float* b, float* c,
    size_t batches, size_t rows, size_t common, size_t cols
) {
    if (batches == 0
        || batches > static_cast<size_t>(INT_MAX)
        || !fitsInt(rows, common, cols)
        || !ready()) {
        return false;
    }

    const size_t a_count = batches * rows * common;
    const size_t b_count = batches * common * cols;
    const size_t c_count = batches * rows * cols;

    if (!buffer_a.ensure(a_count)
        || !buffer_b.ensure(b_count)
        || !buffer_c.ensure(c_count)) {
        return false;
    }

    if (cudaMemcpy(buffer_a.ptr, a, a_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(buffer_b.ptr, b, b_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    const auto batch = static_cast<int>(batches);
    const auto i = static_cast<int>(rows);
    const auto j = static_cast<int>(common);
    const auto k = static_cast<int>(cols);
    dim3 block(1, 16, 16);
    dim3 grid(batch, (k + 15) / 16, (i + 15) / 16);

    cudaTiledBatchGemm<<<grid, block>>>(
        buffer_a.ptr,
        buffer_b.ptr,
        buffer_c.ptr,
        batch, i, j, k
    );
    if (!launchOk()) {
        return false;
    }

    return cudaMemcpy(c, buffer_c.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}
