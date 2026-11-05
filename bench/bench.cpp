#include "NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "NeuralNet/Optimizers/include/AdamW.h"
#include "NumCPP/include/CudaGemm.h"
#include "NumCPP/include/Tensor.h"
#include "Transformer/include/Model.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__)
#include <cpuid.h>
#endif

namespace {

volatile float sink = 0.0f;

std::string cpuBrand() {
#if defined(__GNUC__)
    if (__get_cpuid_max(0x80000000, nullptr) < 0x80000004) {
        return "unknown";
    }

    char brand[49] = {};
    for (unsigned int leaf = 0; leaf < 3; leaf++) {
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        __get_cpuid(0x80000002 + leaf, &eax, &ebx, &ecx, &edx);
        std::memcpy(brand + leaf * 16, &eax, 4);
        std::memcpy(brand + leaf * 16 + 4, &ebx, 4);
        std::memcpy(brand + leaf * 16 + 8, &ecx, 4);
        std::memcpy(brand + leaf * 16 + 12, &edx, 4);
    }

    std::string text(brand);
    const size_t begin = text.find_first_not_of(' ');
    if (begin == std::string::npos) {
        return {};
    }
    const size_t end = text.find_last_not_of(' ');
    return text.substr(begin, end - begin + 1);
#else
    return "unknown";
#endif
}

std::string formatReport(const GemmReport& report) {
    std::string text;
    auto add = [&](const char* name, bool enabled) {
        if (!enabled) {
            return;
        }
        if (!text.empty()) {
            text += " + ";
        }
        text += name;
    };

    add("one thread", report.single_thread);
    add("OpenMP", report.openmp);
    add("cuBLAS", report.cublas);
    add("custom", report.custom);
    if (text.empty()) {
        return "not measured";
    }
    return text;
}

template<class Fn>
double medianMs(Fn&& fn) {
    fn();
    fn();

    std::array<double, 10> samples{};
    for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto stop = std::chrono::steady_clock::now();
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

std::string formatSpeedup(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << "x";
    return out.str();
}

struct Row {
    std::string measurement;
    std::string shape;
    std::string path;
    int threads = 0;
    double one_ms = 0.0;
    double cpu_ms = 0.0;
    bool gpu = false;
    double gpu_ms = 0.0;
    bool custom = false;
    double custom_ms = 0.0;
    double speedup = 0.0;
    std::string cuda_vs_cpu;
    std::string custom_vs_cpu;
};

template<class Fn>
Row timeCase(const std::string& measurement, const std::string& shape, int default_threads, bool gpu_device, Fn&& fn) {
    std::cout << "Timing " << measurement << " " << shape << std::endl;

    Row row;
    row.measurement = measurement;
    row.shape = shape;
    row.threads = default_threads;

    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(false);
    omp_set_num_threads(1);
    row.one_ms = medianMs(fn);
    omp_set_num_threads(default_threads);
    row.cpu_ms = medianMs(fn);

    Tensor::setAllowCudaGemm(true);
    omp_set_num_threads(default_threads);
    Tensor::resetGemmReport();
    fn();
    const GemmReport report = Tensor::gemmReport();
    row.path = formatReport(report);

    if (gpu_device && report.cublas) {
        row.gpu = true;
        row.gpu_ms = medianMs(fn);
    }

    const double used = row.gpu ? row.gpu_ms : row.cpu_ms;
    row.speedup = used > 0.0 ? row.one_ms / used : 0.0;
    if (!row.gpu) {
        row.cuda_vs_cpu = "—";
    } else if (row.gpu_ms < row.cpu_ms) {
        row.cuda_vs_cpu = "faster";
    } else {
        row.cuda_vs_cpu = "not faster";
    }

    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(true);
    Tensor::resetGemmReport();
    fn();
    const GemmReport custom_report = Tensor::gemmReport();
    if (gpu_device && custom_report.custom) {
        row.custom = true;
        row.custom_ms = medianMs(fn);
    }

    Tensor::setAllowCustomGemm(false);
    Tensor::setAllowCudaGemm(true);

    if (!row.custom) {
        row.custom_vs_cpu = "—";
    } else if (row.custom_ms < row.cpu_ms) {
        row.custom_vs_cpu = "faster";
    } else {
        row.custom_vs_cpu = "not faster";
    }

    return row;
}

std::string markdown(const std::string& cpu, int processors, int threads, const std::string& gpu_line, const std::vector<Row>& rows) {
    std::ostringstream out;
    out << "# CPU versus CUDA\n\n";
    out << "Processor: " << cpu << "\n\n";
    out << "Logical processors: " << processors << "\n\n";
    out << "OpenMP threads in the measurement: " << threads << "\n\n";
    out << gpu_line << "\n\n";
    out << "The cuBLAS cutoff is a dimension product of 65536 (`batch * rows * common * cols`). "
        << "A smaller product stays on the CPU, so the GPU column stays empty. "
        << "cuBLAS is marked faster only when its median is lower than the CPU median.\n\n";
    out << "The custom column is the tiled kernel. cuBLAS is turned off for that measurement, "
        << "and the kernel is used for every matmul, including products below 65536. "
        << "Host and device copies are included, as they are in the cuBLAS column.\n\n";
    out << "Each measurement has two warmups, then the median of 10 repeats. "
        << "The CPU column is OpenMP with CUDA disabled. "
        << "One thread is the same CPU code with `omp_set_num_threads(1)`. "
        << "Speedup is the one-thread median divided by the median of the path that actually ran.\n\n";
    out << "Width 32: attention `[1, 64, 8] @ [1, 8, 64]`, feed-forward `[64, 32] @ [32, 128]`, output `[64, 32] @ [32, 5256]`.\n\n";
    out << "Width 128 and 4 heads: attention `[1, 64, 32] @ [1, 32, 64]`, feed-forward `[64, 128] @ [128, 512]`, output `[64, 128] @ [128, 5256]`.\n\n";
    out << "The model has 3 blocks, a context of 64, and a vocabulary of 5256.\n\n";
    out << "| Measurement | Shape | Path | Threads | 1 thread ms | CPU ms | GPU ms | Speedup vs 1 thread | cuBLAS vs CPU | Custom ms | Custom vs CPU |\n";
    out << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";

    for (const Row& row : rows) {
        out << "| " << row.measurement
            << " | " << row.shape
            << " | " << row.path
            << " | " << row.threads
            << " | " << formatMs(row.one_ms)
            << " | " << formatMs(row.cpu_ms)
            << " | " << (row.gpu ? formatMs(row.gpu_ms) : "—")
            << " | " << formatSpeedup(row.speedup)
            << " | " << row.cuda_vs_cpu
            << " | " << (row.custom ? formatMs(row.custom_ms) : "—")
            << " | " << row.custom_vs_cpu
            << " |\n";
    }

    out << "\n";
    return out.str();
}

}

int main() {
    const int default_threads = omp_get_max_threads();
    const int processors = omp_get_num_procs();
#if GPT_HAS_CUDA
    const bool cuda_build = true;
#else
    const bool cuda_build = false;
#endif
    const bool gpu_device = cuda_build && cudaGemmAvailable();

    std::string gpu_line;
    if (!cuda_build) {
        gpu_line = "GPU: nvcc was not found. CUDA rows were left empty.";
    } else if (!gpu_device) {
        gpu_line = "GPU: nvcc was found, but no device is visible. CUDA rows were left empty.";
    } else {
        char label[256] = {};
        if (cudaDeviceLabel(label, sizeof(label))) {
            gpu_line = std::string("GPU: ") + label;
        } else {
            gpu_line = "GPU: cuBLAS is available.";
        }
    }

    std::vector<Row> rows;

    struct GemmShape {
        const char* measurement;
        const char* shape;
        bool batched;
        size_t batch;
        size_t rows_dim;
        size_t common;
        size_t cols;
    };

    const GemmShape shapes[] = {
        {"batchMatmul", "[1, 64, 8] @ [1, 8, 64]", true, 1, 64, 8, 64},
        {"matmul", "[64, 32] @ [32, 128]", false, 1, 64, 32, 128},
        {"matmul", "[64, 32] @ [32, 5256]", false, 1, 64, 32, 5256},
        {"batchMatmul", "[1, 64, 32] @ [1, 32, 64]", true, 1, 64, 32, 64},
        {"matmul", "[64, 128] @ [128, 512]", false, 1, 64, 128, 512},
        {"matmul", "[64, 128] @ [128, 5256]", false, 1, 64, 128, 5256},
    };

    for (const GemmShape& shape : shapes) {
        if (shape.batched) {
            Tensor left({shape.batch, shape.rows_dim, shape.common}, 0.01f);
            Tensor right({shape.batch, shape.common, shape.cols}, 0.01f);
            rows.push_back(timeCase(shape.measurement, shape.shape, default_threads, gpu_device, [&] {
                Tensor result = left.batchMatmul(right);
                sink += result[0];
            }));
        } else {
            Tensor left({shape.rows_dim, shape.common}, 0.01f);
            Tensor right({shape.common, shape.cols}, 0.01f);
            rows.push_back(timeCase(shape.measurement, shape.shape, default_threads, gpu_device, [&] {
                Tensor result = left.matmul(right);
                sink += result[0];
            }));
        }
    }

    constexpr size_t vocab = 5256;
    constexpr size_t block = 64;
    for (size_t width : {size_t{32}, size_t{128}}) {
        const std::string shape = "width " + std::to_string(width) + ", layers 3, block 64, vocab 5256";
        Tensor::seed(1);
        Model model(vocab, block, width, 4, 3, 1, block);
        Tensor input({1, block}, 0.0f);
        Tensor target({1, block, vocab}, 0.0f);
        for (size_t i = 0; i < block; i++) {
            input[i] = static_cast<float>(i % 50);
            target[i * vocab + ((i + 1) % 50)] = 1.0f;
        }

        rows.push_back(timeCase("forward+backward", shape, default_threads, gpu_device, [&] {
            model.zeroGrad();
            Tensor output = model.forward(input);
            sink += output[0];
            model.backward(CrossEntropyLoss::backward(output, target));
        }));

        AdamW optimizer(0.001f, 0.01f);
        rows.push_back(timeCase("full step", shape, default_threads, gpu_device, [&] {
            model.zeroGrad();
            Tensor output = model.forward(input);
            const float loss = CrossEntropyLoss::loss(output, target);
            sink += loss;
            model.backward(CrossEntropyLoss::backward(output, target));
            optimizer.step(model.parameters(), model.gradients());
        }));
    }

    omp_set_num_threads(default_threads);
    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);

    const std::string text = markdown(cpuBrand(), processors, default_threads, gpu_line, rows);
    std::filesystem::create_directories("bench");
    std::ofstream file("bench/RESULTS.md");
    if (!file) {
        throw std::runtime_error("bench/RESULTS.md could not be written");
    }
    file << text;
    std::cout << text;
    std::cerr << "sink " << sink << std::endl;
    return 0;
}
