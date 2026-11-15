#include "Cuda/include/Attention.h"
#include "Cuda/include/RooflineLaunch.h"
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

constexpr double kFp32Peak = 8.1e12;
constexpr double kTensorPeak = 65.0e12;
constexpr double kBandwidth = 320.0e9;
constexpr double kLaunchMs = 0.05;

struct Row {
    std::string kernel;
    std::string shape;
    double ms = 0.0;
    double gflops = 0.0;
    double gbs = 0.0;
    double intensity = 0.0;
    double pct_compute = 0.0;
    double pct_bandwidth = 0.0;
    const char* limit = "launch";
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

Row makeRow(const std::string& kernel, const std::string& shape, double ms, double flops, double bytes, bool tensor) {
    Row row;
    row.kernel = kernel;
    row.shape = shape;
    row.ms = ms;
    const double seconds = ms * 1e-3;
    const double peak = tensor ? kTensorPeak : kFp32Peak;
    const double ridge = peak / kBandwidth;
    if (seconds > 0.0) {
        row.gflops = flops / seconds / 1e9;
        row.gbs = bytes / seconds / 1e9;
        row.pct_compute = 100.0 * (flops / seconds) / peak;
        row.pct_bandwidth = 100.0 * (bytes / seconds) / kBandwidth;
    }
    row.intensity = bytes > 0.0 ? flops / bytes : 0.0;
    if (ms < kLaunchMs) {
        row.limit = "launch";
    } else if (row.intensity < ridge) {
        row.limit = "memory";
    } else {
        row.limit = "compute";
    }
    return row;
}

std::string fixed(double value, int digits) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(digits) << value;
    return out.str();
}

std::string table(const std::vector<Row>& rows) {
    std::string text;
    text += "| Kernel | Shape | ms | GFLOP/s | GB/s | FLOP/byte | % compute | % bandwidth | Limit |\n";
    text += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
    for (const Row& row : rows) {
        text += "| " + row.kernel
            + " | " + row.shape
            + " | " + fixed(row.ms, 3)
            + " | " + fixed(row.gflops, 2)
            + " | " + fixed(row.gbs, 2)
            + " | " + fixed(row.intensity, 2)
            + " | " + fixed(row.pct_compute, 3)
            + " | " + fixed(row.pct_bandwidth, 3)
            + " | " + row.limit
            + " |\n";
    }
    return text;
}

double gemmFlops(size_t rows, size_t common, size_t cols) {
    return 2.0 * static_cast<double>(rows) * static_cast<double>(common) * static_cast<double>(cols);
}

double gemmBytes(size_t rows, size_t common, size_t cols, bool half_inputs) {
    const double a = static_cast<double>(rows) * static_cast<double>(common);
    const double b = static_cast<double>(common) * static_cast<double>(cols);
    const double c = static_cast<double>(rows) * static_cast<double>(cols);
    const double in_bytes = half_inputs ? 2.0 : 4.0;
    return in_bytes * (a + b) + 4.0 * c;
}

double softmaxFlops(size_t heads, size_t length) {
    return 5.0 * static_cast<double>(heads) * static_cast<double>(length) * static_cast<double>(length);
}

double softmaxBytes(size_t heads, size_t length) {
    return 8.0 * static_cast<double>(heads) * static_cast<double>(length) * static_cast<double>(length);
}

double normFlops(size_t length, size_t width) {
    return static_cast<double>(length) * static_cast<double>(width) * (4.0 * static_cast<double>(width) + 8.0);
}

double normBytes(size_t length, size_t width) {
    return 4.0 * (2.0 * static_cast<double>(length) * static_cast<double>(width) + 2.0 * static_cast<double>(width));
}

double attentionFlops(size_t heads, size_t length, size_t d) {
    const double n = static_cast<double>(heads) * static_cast<double>(length) * static_cast<double>(length);
    return n * (4.0 * static_cast<double>(d) + 5.0);
}

double scoredBytes(size_t heads, size_t length, size_t d) {
    const double h = static_cast<double>(heads);
    const double t = static_cast<double>(length);
    const double width = static_cast<double>(d);
    return 4.0 * h * (6.0 * t * width + 4.0 * t * t);
}

double fusedBytes(size_t heads, size_t length, size_t d) {
    const double tiles = std::ceil(static_cast<double>(length) / 64.0);
    const double h = static_cast<double>(heads);
    const double t = static_cast<double>(length);
    const double width = static_cast<double>(d);
    return 4.0 * h * t * width * (2.0 + 2.0 * tiles);
}

#if GPT_HAS_CUDA

struct DeviceFloats {
    float* data = nullptr;

    explicit DeviceFloats(size_t count) {
        if (count == 0 || cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)) != cudaSuccess) {
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
        if (count == 0 || cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(__half)) != cudaSuccess) {
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

bool copyUp(float* device, const std::vector<float>& host) {
    return cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool timeGemm(std::vector<Row>& rows, size_t m, size_t k, size_t n) {
    const size_t a_count = m * k;
    const size_t b_count = k * n;
    const size_t c_count = m * n;
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
    if (!copyUp(a.data, host_a) || !copyUp(b.data, host_b)) {
        return false;
    }
    if (!wmmaQuantizeDevice(a.data, a16.data, a_count) || !wmmaQuantizeDevice(b.data, b16.data, b_count)) {
        return false;
    }

    const std::string shape = std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
    const double flops = gemmFlops(m, k, n);
    bool ok = false;
    const double cublas_ms = medianMs([&] {
        return cudaGemmDevice(a.data, b.data, cublas.data, m, k, n) && cudaDeviceSynchronize() == cudaSuccess;
    }, ok);
    if (!ok) {
        return false;
    }
    const double custom_ms = medianMs([&] {
        return customGemmDevice(a.data, b.data, custom.data, m, k, n) && cudaDeviceSynchronize() == cudaSuccess;
    }, ok);
    if (!ok) {
        return false;
    }
    const double wmma_ms = medianMs([&] {
        return wmmaGemmLaunch(a16.data, b16.data, wmma.data, m, k, n);
    }, ok);
    if (!ok) {
        return false;
    }

    rows.push_back(makeRow("cublas", shape, cublas_ms, flops, gemmBytes(m, k, n, false), false));
    rows.push_back(makeRow("tiled", shape, custom_ms, flops, gemmBytes(m, k, n, false), false));
    rows.push_back(makeRow("wmma", shape, wmma_ms, flops, gemmBytes(m, k, n, true), true));
    return true;
}

bool timeSoftmax(std::vector<Row>& rows, size_t heads, size_t length) {
    const size_t count = heads * length * length;
    std::vector<float> host(count);
    fill(host);
    DeviceFloats scores(count);
    if (scores.data == nullptr || !copyUp(scores.data, host)) {
        return false;
    }
    bool ok = false;
    const double ms = medianMs([&] {
        return rooflineSoftmax(scores.data, heads, length, 0.125f);
    }, ok);
    if (!ok) {
        return false;
    }
    rows.push_back(makeRow(
        "softmax",
        "H=" + std::to_string(heads) + " T=" + std::to_string(length),
        ms,
        softmaxFlops(heads, length),
        softmaxBytes(heads, length),
        false));
    return true;
}

bool timeNorm(std::vector<Row>& rows, size_t length, size_t width) {
    const size_t count = length * width;
    std::vector<float> host(count, 0.1f);
    std::vector<float> gamma(width, 1.0f);
    std::vector<float> beta(width, 0.0f);
    fill(host);
    DeviceFloats x(count);
    DeviceFloats y(count);
    DeviceFloats g(width);
    DeviceFloats b(width);
    if (x.data == nullptr || y.data == nullptr || g.data == nullptr || b.data == nullptr) {
        return false;
    }
    if (!copyUp(x.data, host) || !copyUp(g.data, gamma) || !copyUp(b.data, beta)) {
        return false;
    }
    bool ok = false;
    const double ms = medianMs([&] {
        return rooflineLayerNorm(x.data, y.data, g.data, b.data, length, width);
    }, ok);
    if (!ok) {
        return false;
    }
    rows.push_back(makeRow(
        "layernorm",
        "T=" + std::to_string(length) + " C=" + std::to_string(width),
        ms,
        normFlops(length, width),
        normBytes(length, width),
        false));
    return true;
}

bool timeAttention(std::vector<Row>& rows, size_t heads, size_t length, size_t d) {
    const size_t count = heads * length * d;
    const size_t scores = heads * length * length;
    std::vector<float> host(count);
    fill(host);
    DeviceFloats q(count);
    DeviceFloats k(count);
    DeviceFloats v(count);
    DeviceFloats kt(count);
    DeviceFloats score(scores);
    DeviceFloats scored(count);
    DeviceFloats fused(count);
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr || kt.data == nullptr
        || score.data == nullptr || scored.data == nullptr || fused.data == nullptr) {
        return false;
    }
    if (!copyUp(q.data, host) || !copyUp(k.data, host) || !copyUp(v.data, host)) {
        return false;
    }

    const std::string shape = "H=" + std::to_string(heads) + " T=" + std::to_string(length) + " d=" + std::to_string(d);
    const double flops = attentionFlops(heads, length, d);
    bool ok = false;
    const double scored_ms = medianMs([&] {
        return attentionContextLaunch(
            GemmDevicePath::Cublas, q.data, k.data, v.data, kt.data, score.data, scored.data, heads, length, d);
    }, ok);
    if (!ok) {
        return false;
    }
    const double fused_ms = medianMs([&] {
        return fusedAttentionDevice(q.data, k.data, v.data, fused.data, heads, length, d);
    }, ok);
    if (!ok) {
        return false;
    }
    rows.push_back(makeRow("scored attention", shape, scored_ms, flops, scoredBytes(heads, length, d), false));
    rows.push_back(makeRow("fused attention", shape, fused_ms, flops, fusedBytes(heads, length, d), false));
    return true;
}

#endif

}  // namespace

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. roofline is skipped." << std::endl;
        return 0;
    }

#if !GPT_HAS_CUDA
    std::cout << "CUDA was not part of this build. roofline is skipped." << std::endl;
    return 0;
#else
    char label[128] = {};
    std::string gpu = "GPU: cuBLAS is available.";
    if (cudaDeviceLabel(label, sizeof(label))) {
        gpu = std::string("GPU: ") + label;
    }

    std::vector<Row> rows;
    const size_t gemms[][3] = {
        {64, 32, 128},
        {64, 128, 512},
        {1024, 1024, 1024},
    };
    for (const auto& shape : gemms) {
        std::cout << "Roofline GEMM " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
        if (!timeGemm(rows, shape[0], shape[1], shape[2])) {
            std::cout << "roofline GEMM failed" << std::endl;
            return 1;
        }
    }
    for (size_t length : {size_t{64}, size_t{1024}}) {
        std::cout << "Roofline softmax T=" << length << std::endl;
        if (!timeSoftmax(rows, 4, length)) {
            std::cout << "roofline softmax failed" << std::endl;
            return 1;
        }
    }
    const size_t norms[][2] = {{64, 32}, {64, 128}, {1024, 128}};
    for (const auto& shape : norms) {
        std::cout << "Roofline layernorm T=" << shape[0] << " C=" << shape[1] << std::endl;
        if (!timeNorm(rows, shape[0], shape[1])) {
            std::cout << "roofline layernorm failed" << std::endl;
            return 1;
        }
    }
    const size_t attention[][3] = {{4, 64, 8}, {4, 64, 32}, {4, 1024, 8}, {4, 1024, 32}};
    for (const auto& shape : attention) {
        std::cout << "Roofline attention H=" << shape[0] << " T=" << shape[1] << " d=" << shape[2] << std::endl;
        if (!timeAttention(rows, shape[0], shape[1], shape[2])) {
            std::cout << "roofline attention failed" << std::endl;
            return 1;
        }
    }

    std::string text;
    text += "# Roofline\n\n";
    text += gpu + "\n\n";
    text += "Each cell is two warmups, then the median of 10 repeats. ";
    text += "The compute roofs are the published T4 peaks: 8.1 TFLOP/s for FP32 and 65 TFLOP/s for FP16 tensor cores. ";
    text += "The memory roof is 320 GB/s. ";
    text += "FLOP/byte uses compulsory traffic: each input is read once and each output is written once. ";
    text += "WMMA inputs are FP16. Fused attention reads K and V once for every query tile of 64. ";
    text += "LayerNorm counts the mean and variance once per channel thread, which is the loop the kernel runs. ";
    text += "Softmax counts five operations per score. ";
    text += "A cell under 0.05 ms is marked launch: its percent of peak is not a throughput result. ";
    text += "Above that line, memory means the intensity is below the ridge, and compute means it is above. ";
    text += "These bytes are the algorithm's minimum traffic. Nsight Compute replaces them with measured DRAM bytes when the GPU allows the counters:\n\n";
    text += "`ncu --section SpeedOfLight_Roofline ./build/roofline_bench`\n\n";
    text += table(rows);

    std::cout << "\n" << text;
    std::ofstream file("bench/ROOFLINE.md");
    if (!file) {
        std::cout << "bench/ROOFLINE.md could not be written" << std::endl;
        return 1;
    }
    file << text;
    return file ? 0 : 1;
#endif
}
