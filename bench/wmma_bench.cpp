#include "NumCPP/include/CudaGemm.h"
#include "NumCPP/include/CustomGemm.h"
#include "NumCPP/include/WmmaGemm.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if GPT_HAS_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace {

struct Row {
    size_t rows = 0;
    size_t common = 0;
    size_t cols = 0;
    double cublas_ms = 0.0;
    double custom_ms = 0.0;
    double wmma_ms = 0.0;
    float diff = 0.0f;
};

template<class Fn>
double medianMs(Fn&& fn, bool& ok) {
    ok = static_cast<bool>(fn()) && static_cast<bool>(fn());
    if (!ok) {
        return 0.0;
    }

    std::array<double, 10> samples{};
    for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        ok = static_cast<bool>(fn());
        const auto stop = std::chrono::steady_clock::now();
        if (!ok) {
            return 0.0;
        }
        sample = std::chrono::duration<double, std::milli>(stop - start).count();
    }

    std::sort(samples.begin(), samples.end());
    return 0.5 * (samples[4] + samples[5]);
}

std::string formatMs(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

std::string formatDiff(float value) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(3) << value;
    return out.str();
}

std::string table(const std::vector<Row>& rows) {
    std::string text;
    text += "| Rows | Common | Cols | cuBLAS ms | Custom ms | WMMA ms | Max abs diff |\n";
    text += "| --- | --- | --- | --- | --- | --- | --- |\n";
    for (const Row& row : rows) {
        text += "| " + std::to_string(row.rows)
            + " | " + std::to_string(row.common)
            + " | " + std::to_string(row.cols)
            + " | " + formatMs(row.cublas_ms)
            + " | " + formatMs(row.custom_ms)
            + " | " + formatMs(row.wmma_ms)
            + " | " + formatDiff(row.diff)
            + " |\n";
    }
    return text;
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

struct DeviceHalf {
    __half* data = nullptr;

    explicit DeviceHalf(size_t count) {
        if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(__half)) != cudaSuccess) {
            data = nullptr;
        }
    }

    ~DeviceHalf() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    DeviceHalf(const DeviceHalf&) = delete;
    DeviceHalf& operator=(const DeviceHalf&) = delete;
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

bool measure(size_t rows, size_t common, size_t cols, Row& row) {
    const size_t a_count = rows * common;
    const size_t b_count = common * cols;
    const size_t c_count = rows * cols;
    row.rows = rows;
    row.common = common;
    row.cols = cols;

    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    fill(host_a);
    fill(host_b);

    DeviceFloats a(a_count);
    DeviceFloats b(b_count);
    DeviceFloats cublas(c_count);
    DeviceFloats custom(c_count);
    DeviceFloats wmma(c_count);
    DeviceHalf a16(a_count);
    DeviceHalf b16(b_count);
    if (a.data == nullptr || b.data == nullptr || cublas.data == nullptr || custom.data == nullptr
        || wmma.data == nullptr || a16.data == nullptr || b16.data == nullptr) {
        return false;
    }
    if (cudaMemcpy(a.data, host_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(b.data, host_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (!wmmaQuantizeDevice(a.data, a16.data, a_count) || !wmmaQuantizeDevice(b.data, b16.data, b_count)) {
        return false;
    }

    bool ok = false;
    row.cublas_ms = medianMs([&] {
        return cudaGemmDevice(a.data, b.data, cublas.data, rows, common, cols)
            && cudaDeviceSynchronize() == cudaSuccess;
    }, ok);
    if (!ok) {
        return false;
    }
    row.custom_ms = medianMs([&] {
        return customGemmDevice(a.data, b.data, custom.data, rows, common, cols)
            && cudaDeviceSynchronize() == cudaSuccess;
    }, ok);
    if (!ok) {
        return false;
    }
    row.wmma_ms = medianMs([&] {
        return wmmaGemmLaunch(a16.data, b16.data, wmma.data, rows, common, cols);
    }, ok);
    if (!ok) {
        return false;
    }

    std::vector<float> host_cublas(c_count);
    std::vector<float> host_wmma(c_count);
    if (cudaMemcpy(host_cublas.data(), cublas.data, c_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaMemcpy(host_wmma.data(), wmma.data, c_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    row.diff = maxAbs(host_wmma, host_cublas);
    return true;
}

#endif

}  // namespace

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. tensor-core timing is skipped." << std::endl;
        return 0;
    }

#if !GPT_HAS_CUDA
    std::cout << "CUDA was not part of this build. tensor-core timing is skipped." << std::endl;
    return 0;
#else
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess || major < 7) {
        std::cout << "Tensor cores are not available on this GPU." << std::endl;
        return 1;
    }

    char label[128] = {};
    std::string gpu = "GPU: cuBLAS is available.";
    if (cudaDeviceLabel(label, sizeof(label))) {
        gpu = std::string("GPU: ") + label;
    }

    const size_t shapes[][3] = {
        {16, 16, 16},
        {64, 32, 128},
        {64, 128, 512},
        {64, 32, 64},
    };
    std::vector<Row> rows;
    for (const auto& shape : shapes) {
        std::cout << "Timing tensor-core GEMM " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
        Row row;
        if (!measure(shape[0], shape[1], shape[2], row)) {
            std::cout << "tensor-core timing failed " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
            return 1;
        }
        rows.push_back(row);
    }

    std::string text;
    text += "# Tensor-core GEMM\n\n";
    text += gpu + "\n\n";
    text += "A and B are FP32 and already on the device. Each cell is two warmups, then the median of 10 repeats. ";
    text += "cuBLAS and the tiled kernel run in FP32. WMMA rounds A and B to FP16 before the timer, multiplies on tensor cores, and accumulates in FP32. ";
    text += "The max abs diff is WMMA against cuBLAS on the same inputs. ";
    text += "Training and generate stay on the FP32 path.\n\n";
    text += table(rows);

    std::cout << "\n" << text;

    std::ofstream file("bench/WMMA.md");
    if (!file) {
        std::cout << "bench/WMMA.md could not be written" << std::endl;
        return 1;
    }
    file << text;
    return file ? 0 : 1;
#endif
}
