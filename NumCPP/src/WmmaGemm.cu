#include "../include/WmmaGemm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <climits>

namespace {

constexpr int kTile = 16;

__global__ void quantizeHalf(const float* in, __half* out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

__global__ void wmmaGemm(const __half* a, const __half* b, float* c, int rows, int common, int cols) {
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp = static_cast<int>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
    const int total_warps = static_cast<int>(gridDim.x) * warps_per_block;
    const int tiles_n = cols / kTile;
    const int tiles = (rows / kTile) * tiles_n;

    for (int tile = global_warp; tile < tiles; tile += total_warps) {
        const int tile_m = tile / tiles_n;
        const int tile_n = tile - tile_m * tiles_n;
        const int row = tile_m * kTile;
        const int col = tile_n * kTile;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTile, kTile, kTile, __half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTile, kTile, kTile, __half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTile, kTile, kTile, float> acc;
        nvcuda::wmma::fill_fragment(acc, 0.0f);

        for (int k = 0; k < common; k += kTile) {
            nvcuda::wmma::load_matrix_sync(a_frag, a + static_cast<size_t>(row) * common + k, common);
            nvcuda::wmma::load_matrix_sync(b_frag, b + static_cast<size_t>(k) * cols + col, cols);
            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        nvcuda::wmma::store_matrix_sync(c + static_cast<size_t>(row) * cols + col, acc, cols, nvcuda::wmma::mem_row_major);
    }
}

bool fitsTile(size_t rows, size_t common, size_t cols) {
    return rows != 0 && common != 0 && cols != 0
        && rows % kTile == 0 && common % kTile == 0 && cols % kTile == 0
        && rows <= static_cast<size_t>(INT_MAX)
        && common <= static_cast<size_t>(INT_MAX)
        && cols <= static_cast<size_t>(INT_MAX);
}

}  // namespace

bool wmmaGemmDevice(const float* a, const float* b, float* c, size_t rows, size_t common, size_t cols) {
    if (a == nullptr || b == nullptr || c == nullptr || !fitsTile(rows, common, cols)) {
        return false;
    }

    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess || major < 7) {
        return false;
    }

    const size_t a_count = rows * common;
    const size_t b_count = common * cols;
    __half* a16 = nullptr;
    __half* b16 = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&a16), a_count * sizeof(__half)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&b16), b_count * sizeof(__half)) != cudaSuccess) {
        cudaFree(a16);
        return false;
    }

    constexpr int kThreads = 256;
    quantizeHalf<<<static_cast<unsigned>((a_count + kThreads - 1) / kThreads), kThreads>>>(a, a16, a_count);
    quantizeHalf<<<static_cast<unsigned>((b_count + kThreads - 1) / kThreads), kThreads>>>(b, b16, b_count);

    constexpr int kWarpsPerBlock = 4;
    const int tiles = static_cast<int>((rows / kTile) * (cols / kTile));
    int blocks = (tiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > 128) {
        blocks = 128;
    }
    wmmaGemm<<<blocks, kWarpsPerBlock * 32>>>(a16, b16, c, static_cast<int>(rows), static_cast<int>(common), static_cast<int>(cols));

    const bool ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    cudaFree(b16);
    cudaFree(a16);
    return ok;
}
