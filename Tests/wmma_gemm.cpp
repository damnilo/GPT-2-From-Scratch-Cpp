#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/WmmaGemm.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if GPT_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

constexpr float kAtol = 1e-2f;

void report(const std::string& name, bool ok, float diff, int& failures) {
    std::cout << name << (ok ? " correct" : " failed");
    if (!ok) {
        std::cout << " (max abs diff " << diff << ")";
        ++failures;
    } else if (diff == diff) {
        std::cout << " (max abs diff " << diff << ")";
    }
    std::cout << std::endl;
}

#if GPT_HAS_CUDA

struct DeviceFloats {
    float* data = nullptr;

    explicit DeviceFloats(size_t count) {
        if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)) != cudaSuccess) {
            data = nullptr;
        }
    }

    ~DeviceFloats() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;
};

void fill(std::vector<float>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i + 1)) * 0.1f;
    }
}

float maxAbs(const std::vector<float>& got, const std::vector<float>& expected) {
    if (got.size() != expected.size()) {
        return INFINITY;
    }
    float diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::fabs(got[i] - expected[i]));
    }
    return diff;
}

float checkCase(size_t rows, size_t common, size_t cols) {
    const size_t a_count = rows * common;
    const size_t b_count = common * cols;
    const size_t c_count = rows * cols;
    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    fill(host_a);
    fill(host_b);

    DeviceFloats a(a_count);
    DeviceFloats b(b_count);
    DeviceFloats reference(c_count);
    DeviceFloats wmma(c_count);
    if (a.data == nullptr || b.data == nullptr || reference.data == nullptr || wmma.data == nullptr) {
        return INFINITY;
    }
    if (cudaMemcpy(a.data, host_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(b.data, host_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return INFINITY;
    }
    if (!cudaGemmDevice(a.data, b.data, reference.data, rows, common, cols)
        || cudaDeviceSynchronize() != cudaSuccess
        || !wmmaGemmDevice(a.data, b.data, wmma.data, rows, common, cols)) {
        return INFINITY;
    }

    std::vector<float> host_reference(c_count);
    std::vector<float> host_wmma(c_count);
    if (cudaMemcpy(host_reference.data(), reference.data, c_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaMemcpy(host_wmma.data(), wmma.data, c_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return INFINITY;
    }
    return maxAbs(host_wmma, host_reference);
}

#endif

}  // namespace

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. tensor-core GEMM is skipped." << std::endl;
        return 0;
    }

#if !GPT_HAS_CUDA
    std::cout << "CUDA was not part of this build. tensor-core GEMM is skipped." << std::endl;
    return 0;
#else
    int failures = 0;
    const size_t shapes[][3] = {
        {16, 16, 16},
        {64, 32, 128},
        {64, 128, 512},
        {64, 32, 64},
    };
    for (const auto& shape : shapes) {
        const size_t rows = shape[0];
        const size_t common = shape[1];
        const size_t cols = shape[2];
        const float diff = checkCase(rows, common, cols);
        const std::string name = "wmma " + std::to_string(rows) + "x" + std::to_string(common) + "x" + std::to_string(cols);
        report(name, diff <= kAtol, diff, failures);
    }

    DeviceFloats dummy(1);
    const bool rejected = dummy.data != nullptr
        && !wmmaGemmDevice(dummy.data, dummy.data, dummy.data, 64, 8, 64);
    std::cout << "wmma rejects common=8 " << (rejected ? "correct" : "failed") << std::endl;
    if (!rejected) {
        ++failures;
    }
    return failures == 0 ? 0 : 1;
#endif
}
