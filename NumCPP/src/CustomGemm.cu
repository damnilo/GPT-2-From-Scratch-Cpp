#include "../include/CustomGemm.h"

#include <cuda_runtime.h>

#include <climits>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void cudaTiledGemm(const float* a, const float* b, float* c,
    int rows, int commons, int cols) {

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * 16 + tx;
    
    int rowBase = blockIdx.y * BM;
    int colBase = blockIdx.x * BN;
    int row0 = rowBase + ty * TM;
    int col0 = colBase + tx * TN;
    
    __shared__ float A[BM][BK];
    __shared__ float B[BK][BN];
    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    for (int i = 0; i < commons; i += BK) {

        int r = tid / 4;
        int c = (tid % 4) * 4;

        for (int j = 0; j < 4; ++j) {
            if ((rowBase + r) < rows && (i + c + j) < commons) {
                A[r][c + j] = a[(rowBase + r) * commons + (i + c + j)];
            } else {
                A[r][c + j] = 0;
            }
        }

        r = tid / 16;
        c = (tid % 16) * 4;

        for (int j = 0; j < 4; ++j) {
            if ((colBase + c + j) < cols && (i + r) < commons) {
                B[r][c + j] = b[(i + r) * cols + (colBase + c + j)];
            } else {
                B[r][c + j] = 0;
            }
        }

        __syncthreads();

        for (int j = 0; j < BK; ++j) {
            for (int k = 0; k < TM; ++k) {
                a_reg[k] = A[ty * TM + k][j];
            }
            for (int k = 0; k < TN; ++k) {
                b_reg[k] = B[j][tx * TN + k];
            }
            for (int k = 0; k < TM; ++k) {
                for (int l = 0; l < TN; ++l) {
                    acc[k][l] += a_reg[k] * b_reg[l];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            if ((row0 + i) < rows && (col0 + j) < cols) {
                c[(row0 + i) * cols + col0 + j] = acc[i][j];
            }
        }
    }
}

__global__ void cudaTiledGemmFast(
    const float* a, const float* b, float* c,
    int rows, int commons, int cols
) {
    (void)rows;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;
    const int row0 = rowBase + threadIdx.y * TM;
    const int col0 = colBase + threadIdx.x * TN;
    const int tid = threadIdx.y * 16 + threadIdx.x;

    const int a_row = tid / 4;
    const int a_col = (tid % 4) * 4;
    const int b_row = tid / 16;
    const int b_col = (tid % 16) * 4;

    __shared__ __align__(16) float A[2][BM][BK];
    __shared__ __align__(16) float B[2][BK][BN];
    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    const int a_index0 = (rowBase + a_row) * commons + a_col;
    *reinterpret_cast<float4*>(&A[0][a_row][a_col]) = *reinterpret_cast<const float4*>(a + a_index0);
    const int b_index0 = b_row * cols + (colBase + b_col);
    *reinterpret_cast<float4*>(&B[0][b_row][b_col]) = *reinterpret_cast<const float4*>(b + b_index0);
    __syncthreads();

    int stage = 0;
    for (int k = 0; k < commons; k += BK) {
        const int next = stage ^ 1;
        if (k + BK < commons) {
            const int a_index = (rowBase + a_row) * commons + (k + BK + a_col);
            *reinterpret_cast<float4*>(&A[next][a_row][a_col]) = *reinterpret_cast<const float4*>(a + a_index);
            const int b_index = (k + BK + b_row) * cols + (colBase + b_col);
            *reinterpret_cast<float4*>(&B[next][b_row][b_col]) = *reinterpret_cast<const float4*>(b + b_index);
        }

        for (int j = 0; j < BK; ++j) {
            for (int element = 0; element < TM; ++element) {
                a_reg[element] = A[stage][threadIdx.y * TM + element][j];
            }
            for (int element = 0; element < TN; ++element) {
                b_reg[element] = B[stage][j][threadIdx.x * TN + element];
            }
            for (int row = 0; row < TM; ++row) {
                for (int col = 0; col < TN; ++col) {
                    acc[row][col] += a_reg[row] * b_reg[col];
                }
            }
        }

        __syncthreads();
        stage = next;
    }

    for (int row = 0; row < TM; ++row) {
        float4 out;
        out.x = acc[row][0];
        out.y = acc[row][1];
        out.z = acc[row][2];
        out.w = acc[row][3];
        *reinterpret_cast<float4*>(c + (row0 + row) * cols + col0) = out;
    }
}

__global__ void cudaTiledBatchGemm(
    const float* a, const float* b, float* c,
    int batches, int rows, int commons, int cols
) {

    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 16 + ty;

    int colBase = blockIdx.y * BN;
    int rowBase = blockIdx.z * BM;

    int row0 = rowBase + tz * TM;
    int col0 = colBase + ty * TN;

    int batch = blockIdx.x;

    if (batch >= batches) {
        return;
    }

    const float* a_batch = a + batch * rows * commons;
    const float* b_batch = b + batch * commons * cols;
    float* c_batch = c + batch * rows * cols;

    __shared__ float A[BM][BK];
    __shared__ float B[BK][BN];
    
    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    for (int i = 0; i < commons; i += BK) {
        
        int r = tid / 4;
        int c = (tid % 4) * 4;

        for (int j = 0; j < 4; ++j) {
            if ((rowBase + r) < rows && (c + i + j) < commons) {
                A[r][c + j] = a_batch[(rowBase + r) * commons + (c + i + j)];
            } else {
                A[r][c + j] = 0.0f;
            }
        }

        r = tid / 16;
        c = (tid % 16) * 4;

        for (int j = 0; j < 4; ++j) {
            if ((colBase + c + j) < cols && (r + i) < commons) {
                B[r][c + j] = b_batch[(r + i) * cols + colBase + c + j];
            } else {
                B[r][c + j] = 0.0f;
            }
        }

        __syncthreads();

        for (int j = 0; j < BK; ++j) {
            for (int k = 0; k < TM; ++k) {
                a_reg[k] = A[tz * TM + k][j];
            }
            for (int k = 0; k < TN; ++k) {
                b_reg[k] = B[j][ty * TN + k];
            }
            for (int k = 0; k < TM; ++k) {
                for (int l = 0; l < TN; ++l) {
                    acc[k][l] += a_reg[k] * b_reg[l];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            if ((row0 + i) < rows && (col0 + j) < cols) {
                c_batch[(row0 + i) * cols + (col0 + j)] = acc[i][j];
            }
        }
    }
}

__global__ void cudaTiledBatchGemmFast(
    const float* a, const float* b, float* c,
    int batches, int rows, int commons, int cols
) {
    const int batch = blockIdx.x;
    if (batch >= batches) {
        return;
    }
    (void)rows;

    const float* a_batch = a + static_cast<size_t>(batch) * rows * commons;
    const float* b_batch = b + static_cast<size_t>(batch) * commons * cols;
    float* c_batch = c + static_cast<size_t>(batch) * rows * cols;

    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tz * 16 + ty;
    const int rowBase = blockIdx.z * BM;
    const int colBase = blockIdx.y * BN;
    const int row0 = rowBase + tz * TM;
    const int col0 = colBase + ty * TN;

    const int a_row = tid / 4;
    const int a_col = (tid % 4) * 4;
    const int b_row = tid / 16;
    const int b_col = (tid % 16) * 4;

    __shared__ __align__(16) float A[2][BM][BK];
    __shared__ __align__(16) float B[2][BK][BN];
    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    const int a_index0 = (rowBase + a_row) * commons + a_col;
    *reinterpret_cast<float4*>(&A[0][a_row][a_col]) = *reinterpret_cast<const float4*>(a_batch + a_index0);
    const int b_index0 = b_row * cols + (colBase + b_col);
    *reinterpret_cast<float4*>(&B[0][b_row][b_col]) = *reinterpret_cast<const float4*>(b_batch + b_index0);
    __syncthreads();

    int stage = 0;
    for (int k = 0; k < commons; k += BK) {
        const int next = stage ^ 1;
        if (k + BK < commons) {
            const int a_index = (rowBase + a_row) * commons + (k + BK + a_col);
            *reinterpret_cast<float4*>(&A[next][a_row][a_col]) = *reinterpret_cast<const float4*>(a_batch + a_index);
            const int b_index = (k + BK + b_row) * cols + (colBase + b_col);
            *reinterpret_cast<float4*>(&B[next][b_row][b_col]) = *reinterpret_cast<const float4*>(b_batch + b_index);
        }

        for (int j = 0; j < BK; ++j) {
            for (int element = 0; element < TM; ++element) {
                a_reg[element] = A[stage][tz * TM + element][j];
            }
            for (int element = 0; element < TN; ++element) {
                b_reg[element] = B[stage][j][ty * TN + element];
            }
            for (int row = 0; row < TM; ++row) {
                for (int col = 0; col < TN; ++col) {
                    acc[row][col] += a_reg[row] * b_reg[col];
                }
            }
        }

        __syncthreads();
        stage = next;
    }

    for (int row = 0; row < TM; ++row) {
        float4 out;
        out.x = acc[row][0];
        out.y = acc[row][1];
        out.z = acc[row][2];
        out.w = acc[row][3];
        *reinterpret_cast<float4*>(c_batch + (row0 + row) * cols + col0) = out;
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

bool fastTile(int rows, int commons, int cols) {
    return rows % BM == 0 && cols % BN == 0 && commons % BK == 0;
}

void launchGemm(
    const float* a, const float* b, float* c,
    int batches, int rows, int commons, int cols
) {
    if (batches == 1) {
        const dim3 block(16, 16);
        const dim3 grid((cols + BN - 1) / BN, (rows + BM - 1) / BM);
        if (fastTile(rows, commons, cols)) {
            cudaTiledGemmFast<<<grid, block>>>(a, b, c, rows, commons, cols);
        } else {
            cudaTiledGemm<<<grid, block>>>(a, b, c, rows, commons, cols);
        }
        return;
    }

    const dim3 block(1, 16, 16);
    const dim3 grid(batches, (cols + BN - 1) / BN, (rows + BM - 1) / BM);
    if (fastTile(rows, commons, cols)) {
        cudaTiledBatchGemmFast<<<grid, block>>>(a, b, c, batches, rows, commons, cols);
    } else {
        cudaTiledBatchGemm<<<grid, block>>>(a, b, c, batches, rows, commons, cols);
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
    launchGemm(buffer_a.ptr, buffer_b.ptr, buffer_c.ptr, 1, i, j, k);
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
    launchGemm(buffer_a.ptr, buffer_b.ptr, buffer_c.ptr, batch, i, j, k);
    if (!launchOk()) {
        return false;
    }

    return cudaMemcpy(c, buffer_c.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool customGemmDevice(const float* a, const float* b, float* c,
    size_t rows, size_t common, size_t cols
) {
    if (!fitsInt(rows, common, cols) || !ready()) {
        return false;
    }

    const auto i = static_cast<int>(rows);
    const auto j = static_cast<int>(common);
    const auto k = static_cast<int>(cols);
    launchGemm(a, b, c, 1, i, j, k);
    return launchOk();
}

bool customGemmBatchDevice(
    const float* a, const float* b, float* c,
    size_t batches, size_t rows, size_t commons, size_t cols
) {
    if (!fitsInt(rows, commons, cols) || !ready() || batches == 0 || batches > INT_MAX) {
        return false;
    }
    
    const auto i = static_cast<int>(rows);
    const auto j = static_cast<int>(commons);
    const auto k = static_cast<int>(cols);
    const auto batch = static_cast<int>(batches);
    launchGemm(a, b, c, batch, i, j, k);
    return launchOk();
}