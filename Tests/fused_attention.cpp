#include "../Cuda/include/Attention.h"
#include "../NumCPP/include/CudaGemm.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if GPT_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

constexpr float kAtol = 1e-3f;
constexpr size_t kHeads = 4;

void report(const std::string& name, bool ok, float diff, int& failures) {
    std::cout << name << (ok ? " correct" : " failed");
    if (!ok) {
        std::cout << " (max abs diff " << diff << ")";
        ++failures;
    }
    std::cout << std::endl;
}

void fill(std::vector<float>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i) * 17) % 100) / 50.0f - 1.0f;
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

float checkCase(size_t width, size_t length, GemmDevicePath path) {
    const size_t d = width / kHeads;
    const size_t count = kHeads * length * d;
    std::vector<float> host_q(count);
    std::vector<float> host_k(count);
    std::vector<float> host_v(count);
    fill(host_q);
    fill(host_k);
    fill(host_v);
    for (size_t i = 0; i < host_k.size(); ++i) {
        host_k[i] += 0.1f;
        host_v[i] -= 0.2f;
    }

    DeviceFloats q(count);
    DeviceFloats k(count);
    DeviceFloats v(count);
    DeviceFloats reference(count);
    DeviceFloats fused(count);
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr
        || reference.data == nullptr || fused.data == nullptr) {
        return INFINITY;
    }
    if (cudaMemcpy(q.data, host_q.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(k.data, host_k.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(v.data, host_v.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return INFINITY;
    }
    if (!attentionContextDevice(path, q.data, k.data, v.data, reference.data, kHeads, length, d)
        || !fusedAttentionDevice(q.data, k.data, v.data, fused.data, kHeads, length, d)) {
        return INFINITY;
    }

    std::vector<float> host_reference(count);
    std::vector<float> host_fused(count);
    if (cudaMemcpy(host_reference.data(), reference.data, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaMemcpy(host_fused.data(), fused.data, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return INFINITY;
    }
    return maxAbs(host_fused, host_reference);
}

#endif

}

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. fused attention is skipped." << std::endl;
        return 0;
    }

#if !GPT_HAS_CUDA
    std::cout << "CUDA was not part of this build. fused attention is skipped." << std::endl;
    return 0;
#else
    int failures = 0;
    for (size_t width : {size_t{32}, size_t{128}}) {
        for (size_t length : {size_t{8}, size_t{64}}) {
            const std::string shape = " width=" + std::to_string(width) + " T=" + std::to_string(length);
            const float cublas = checkCase(width, length, GemmDevicePath::Cublas);
            report("cublas" + shape, cublas <= kAtol, cublas, failures);
            const float custom = checkCase(width, length, GemmDevicePath::Custom);
            report("custom" + shape, custom <= kAtol, custom, failures);
        }
    }
    return failures == 0 ? 0 : 1;
#endif
}
