#include "Cuda/include/Attention.h"
#include "NumCPP/include/CudaGemm.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if GPT_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

constexpr size_t kHeads = 4;

struct Row {
    size_t d = 0;
    size_t T = 0;
    double cublas_ms = 0.0;
    double custom_ms = 0.0;
    double fused_ms = 0.0;
    size_t score_bytes = 0;
    size_t kt_bytes = 0;
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

std::string table(const std::vector<Row>& rows) {
    std::string text;
    text += "| Head width | T | cuBLAS ms | Custom ms | Fused ms | Score bytes | Transposed K bytes | Extra bytes |\n";
    text += "| --- | --- | --- | --- | --- | --- | --- | --- |\n";
    for (const Row& row : rows) {
        text += "| " + std::to_string(row.d)
            + " | " + std::to_string(row.T)
            + " | " + formatMs(row.cublas_ms)
            + " | " + formatMs(row.custom_ms)
            + " | " + formatMs(row.fused_ms)
            + " | " + std::to_string(row.score_bytes)
            + " | " + std::to_string(row.kt_bytes)
            + " | " + std::to_string(row.score_bytes + row.kt_bytes)
            + " |\n";
    }
    return text;
}

#if GPT_HAS_CUDA

struct DeviceFloats {
    float* data = nullptr;

    explicit DeviceFloats(size_t bytes) {
        if (bytes == 0 || cudaMalloc(reinterpret_cast<void**>(&data), bytes) != cudaSuccess) {
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
        data[i] = static_cast<float>((static_cast<int>(i) * 17) % 100) / 50.0f - 1.0f;
    }
}

bool measure(size_t d, size_t T, Row& row) {
    const size_t count = kHeads * T * d;
    const size_t score_count = kHeads * T * T;
    row.d = d;
    row.T = T;
    row.score_bytes = score_count * sizeof(float);
    row.kt_bytes = count * sizeof(float);

    std::vector<float> host(count);
    fill(host);

    DeviceFloats q(count * sizeof(float));
    DeviceFloats k(count * sizeof(float));
    DeviceFloats v(count * sizeof(float));
    DeviceFloats kt(row.kt_bytes);
    DeviceFloats scores(row.score_bytes);
    DeviceFloats scored(count * sizeof(float));
    DeviceFloats fused(count * sizeof(float));
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr || kt.data == nullptr
        || scores.data == nullptr || scored.data == nullptr || fused.data == nullptr) {
        return false;
    }
    if (cudaMemcpy(q.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(k.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(v.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    bool ok = false;
    row.cublas_ms = medianMs([&] {
        return attentionContextLaunch(
            GemmDevicePath::Cublas, q.data, k.data, v.data, kt.data, scores.data, scored.data, kHeads, T, d);
    }, ok);
    if (!ok) {
        return false;
    }
    row.custom_ms = medianMs([&] {
        return attentionContextLaunch(
            GemmDevicePath::Custom, q.data, k.data, v.data, kt.data, scores.data, scored.data, kHeads, T, d);
    }, ok);
    if (!ok) {
        return false;
    }
    row.fused_ms = medianMs([&] {
        return fusedAttentionDevice(q.data, k.data, v.data, fused.data, kHeads, T, d);
    }, ok);
    return ok;
}

#endif

}

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. fused attention timing is skipped." << std::endl;
        return 0;
    }

#if !GPT_HAS_CUDA
    std::cout << "CUDA was not part of this build. fused attention timing is skipped." << std::endl;
    return 0;
#else
    char label[128] = {};
    std::string gpu = "GPU: cuBLAS is available.";
    if (cudaDeviceLabel(label, sizeof(label))) {
        gpu = std::string("GPU: ") + label;
    }

    const size_t widths[] = {8, 32};
    const size_t lengths[] = {64, 128, 256, 512, 1024};
    std::vector<Row> rows;
    for (size_t d : widths) {
        for (size_t T : lengths) {
            std::cout << "Timing fused attention d=" << d << " T=" << T << std::endl;
            Row row;
            if (!measure(d, T, row)) {
                std::cout << "fused attention timing failed d=" << d << " T=" << T << std::endl;
                return 1;
            }
            rows.push_back(row);
        }
    }

    std::string text;
    text += "# Fused attention\n\n";
    text += gpu + "\n\n";
    text += "Q, K, and V are already on the device. Each cell is two warmups, then the median of 10 repeats. ";
    text += "The scored path is a transposed K, a product, a masked softmax, and a product with V. ";
    text += "Its scratch, the transposed K and the score matrix, is allocated before the timer. ";
    text += "The fused path does not allocate that scratch. Four heads. ";
    text += "Training, generate, and the context of 64 stay on their current path.\n\n";
    text += table(rows);

    std::cout << "\n" << text;

    std::ofstream file("bench/FUSED.md");
    if (!file) {
        std::cout << "bench/FUSED.md could not be written" << std::endl;
        return 1;
    }
    file << text;
    return file ? 0 : 1;
#endif
}
