#include "NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "NeuralNet/Optimizers/include/AdamW.h"
#include "NumCPP/include/CudaGemm.h"
#include "NumCPP/include/Tensor.h"
#include "Transformer/include/Model.h"
#include "train.h"
#include "Cuda/include/Attention.h"
#include "Cuda/include/FeedForward.h"
#include "Cuda/include/Vocab.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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

std::string formatRate(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}

struct HybridRow {
    std::string shape;
    bool gpu = false;
    double cublas_ms = 0.0;
    double custom_ms = 0.0;
};

double hybridStepMs(size_t width, GemmDevicePath path) {
    constexpr size_t vocab = 5256;
    constexpr size_t block = 64;
    Tensor::seed(1);
    Model model(vocab, block, width, 4, 3, 1, block);
    AdamW optimizer(0.001f, 0.01f);
    Tensor input({1, block}, 0.0f);
    Tensor target({1, block, vocab}, 0.0f);
    for (size_t i = 0; i < block; ++i) {
        input[i] = static_cast<float>(i % 50);
        target[i * vocab + ((i + 1) % 50)] = 1.0f;
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);

    std::vector<AttentionDevice> slots;
    if (!enableDeviceAttention(model, path, slots)) {
        throw std::runtime_error("device attention setup failed");
    }
    const double milliseconds = medianMs([&] {
        model.zeroGrad();
        const Tensor output = model.forward(input);
        sink += CrossEntropyLoss::loss(output, target);
        model.backward(CrossEntropyLoss::backward(output, target));
        optimizer.step(model.parameters(), model.gradients());
        if (!refreshAttentionWeights(model, slots)) {
            throw std::runtime_error("attention weight refresh failed");
        }
    });
    disableDeviceAttention(model, slots);
    return milliseconds;
}

double blockStepMs(size_t width, GemmDevicePath path) {
    constexpr size_t vocab = 5256;
    constexpr size_t block = 64;
    Tensor::seed(1);
    Model model(vocab, block, width, 4, 3, 1, block);
    AdamW optimizer(0.001f, 0.01f);
    Tensor input({1, block}, 0.0f);
    Tensor target({1, block, vocab}, 0.0f);
    for (size_t i = 0; i < block; ++i) {
        input[i] = static_cast<float>(i % 50);
        target[i * vocab + ((i + 1) % 50)] = 1.0f;
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);

    std::vector<AttentionDevice> attention_slots;
    std::vector<FeedForwardDevice> feed_slots;
    VocabDevice vocab_slot;
    if (!enableDeviceAttention(model, path, attention_slots)
        || !enableDeviceFeedForward(model, path, feed_slots)
        || !enableDeviceVocab(model, path, vocab_slot)) {
        disableDeviceVocab(model, vocab_slot);
        disableDeviceFeedForward(model, feed_slots);
        disableDeviceAttention(model, attention_slots);
        throw std::runtime_error("device block setup failed");
    }
    const double milliseconds = medianMs([&] {
        model.zeroGrad();
        const Tensor output = model.forward(input);
        sink += CrossEntropyLoss::loss(output, target);
        model.backward(CrossEntropyLoss::backward(output, target));
        optimizer.step(model.parameters(), model.gradients());
        if (!refreshAttentionWeights(model, attention_slots)
            || !refreshFeedForwardWeights(model, feed_slots)
            || !refreshVocabWeights(model, vocab_slot)) {
            throw std::runtime_error("weight refresh failed");
        }
    });
    disableDeviceVocab(model, vocab_slot);
    disableDeviceFeedForward(model, feed_slots);
    disableDeviceAttention(model, attention_slots);
    return milliseconds;
}

struct AttentionRow {
    std::string shape;
    double cpu_ms = 0.0;
    bool gpu = false;
    double cublas_ms = 0.0;
    double custom_ms = 0.0;
    float cublas_diff = 0.0f;
    float custom_diff = 0.0f;
};

float vectorDiff(const std::vector<float>& got, const Tensor& expected) {
    if (got.size() != expected.size()) {
        return INFINITY;
    }
    float diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::fabs(got[i] - expected[i]));
    }
    return diff;
}

float attentionDiff(MultiHeadAttention& layer, const Tensor& input_grad, const AttentionGrad& got) {
    float diff = vectorDiff(got.input, input_grad);
    const std::vector<Tensor*> grads = layer.gradients();
    if (got.heads.size() * 8 + 2 != grads.size()) {
        return INFINITY;
    }
    for (size_t h = 0; h < got.heads.size(); ++h) {
        const size_t base = h * 8;
        const HeadGrad& head = got.heads[h];
        diff = std::max(diff, vectorDiff(head.Wq, *grads[base]));
        diff = std::max(diff, vectorDiff(head.bq, *grads[base + 1]));
        diff = std::max(diff, vectorDiff(head.Wk, *grads[base + 2]));
        diff = std::max(diff, vectorDiff(head.bk, *grads[base + 3]));
        diff = std::max(diff, vectorDiff(head.Wv, *grads[base + 4]));
        diff = std::max(diff, vectorDiff(head.bv, *grads[base + 5]));
        diff = std::max(diff, vectorDiff(head.Wo, *grads[base + 6]));
        diff = std::max(diff, vectorDiff(head.bo, *grads[base + 7]));
    }
    diff = std::max(diff, vectorDiff(got.Wo, *grads[grads.size() - 2]));
    diff = std::max(diff, vectorDiff(got.bo, *grads[grads.size() - 1]));
    return diff;
}

std::string formatDiff(float value) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(3) << value;
    return out.str();
}

struct GenerateRow {
    std::string shape;
    double cpu_per_s = 0.0;
    bool gpu = false;
    double cublas_per_s = 0.0;
    double custom_per_s = 0.0;
};

double tokensPerSecond(double milliseconds, size_t count) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    return 1000.0 * static_cast<double>(count) / milliseconds;
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

std::string markdown(const std::string& cpu, int processors, int threads, const std::string& gpu_line, const std::vector<Row>& rows, const std::vector<GenerateRow>& generated, const std::vector<AttentionRow>& attention, const std::vector<HybridRow>& hybrid, const std::vector<HybridRow>& block) {
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
    out << "Generation appends 64 tokens to a context that is already 64 tokens long. "
        << "Each path uses two warmups and the median of 10 repeats. "
        << "Device weights are uploaded once, outside the timer. "
        << "The CPU path still projects and applies softmax to every position. "
        << "The device paths project and apply softmax only to the last position.\n\n";
    out << "| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |\n";
    out << "| --- | --- | --- | --- |\n";
    for (const GenerateRow& row : generated) {
        out << "| " << row.shape
            << " | " << formatRate(row.cpu_per_s)
            << " | " << (row.gpu ? formatRate(row.cublas_per_s) : "—")
            << " | " << (row.gpu ? formatRate(row.custom_per_s) : "—")
            << " |\n";
    }

    out << "\n";
    out << "Attention backward is one module, 4 heads, and a context of 64. "
        << "The CPU forward and the device forward state are outside the timer. "
        << "The difference is the largest absolute gap against the CPU backward. "
        << "A gap above 1e-3 is a failed comparison.\n\n";
    out << "| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |\n";
    out << "| --- | --- | --- | --- | --- | --- |\n";
    for (const AttentionRow& row : attention) {
        out << "| " << row.shape
            << " | " << formatMs(row.cpu_ms)
            << " | " << (row.gpu ? formatMs(row.cublas_ms) : "—")
            << " | " << (row.gpu ? formatMs(row.custom_ms) : "—")
            << " | " << (row.gpu ? formatDiff(row.cublas_diff) : "—")
            << " | " << (row.gpu ? formatDiff(row.custom_diff) : "—")
            << " |\n";
    }

    out << "\n";
    out << "Training step with device attention. "
        << "LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU. "
        << "Attention weights are refreshed after AdamW. "
        << "The other products keep the cuBLAS rule.\n\n";
    out << "| Shape | cuBLAS ms | Custom ms |\n";
    out << "| --- | --- | --- |\n";
    for (const HybridRow& row : hybrid) {
        out << "| " << row.shape
            << " | " << (row.gpu ? formatMs(row.cublas_ms) : "—")
            << " | " << (row.gpu ? formatMs(row.custom_ms) : "—")
            << " |\n";
    }

    out << "\n";
    out << "Training step with device attention, device feed-forward, and the device vocabulary projection. "
        << "LayerNorm, the embeddings, and AdamW stay on the CPU. "
        << "Attention, feed-forward, and vocabulary weights are refreshed after AdamW. "
        << "The other products keep the cuBLAS rule.\n\n";
    out << "| Shape | cuBLAS ms | Custom ms |\n";
    out << "| --- | --- | --- |\n";
    for (const HybridRow& row : block) {
        out << "| " << row.shape
            << " | " << (row.gpu ? formatMs(row.cublas_ms) : "—")
            << " | " << (row.gpu ? formatMs(row.custom_ms) : "—")
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
    std::vector<GenerateRow> generated;
    std::vector<AttentionRow> attention;
    int attention_failures = 0;

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

        std::vector<int> context(block);
        for (size_t i = 0; i < block; ++i) {
            context[i] = static_cast<int>(i % 50);
        }
        constexpr size_t new_tokens = 64;
        constexpr float temperature = 0.8f;

        std::cout << "Timing generate " << shape << std::endl;
        Tensor::setAllowCudaGemm(false);
        Tensor::setAllowCustomGemm(false);
        omp_set_num_threads(default_threads);

        GenerateRow generation;
        generation.shape = shape;
        const double cpu_ms = medianMs([&] {
            std::mt19937 rng(1);
            const std::vector<int> out = generate(model, context, new_tokens, temperature, rng);
            sink += static_cast<float>(out.back());
        });
        generation.cpu_per_s = tokensPerSecond(cpu_ms, new_tokens);

        if (gpu_device) {
            DeviceWeights weights;
            DeviceWorkspace workspace;
            if (!uploadDeviceWeights(model, weights) || !allocDeviceWorkspace(model.getConfig(), workspace)) {
                throw std::runtime_error("device generate setup failed");
            }
            const double cublas_ms = medianMs([&] {
                std::mt19937 rng(1);
                const std::vector<int> out = generateDevice(GemmDevicePath::Cublas, weights, workspace, model, context, new_tokens, temperature, rng);
                sink += static_cast<float>(out.back());
            });
            const double custom_ms = medianMs([&] {
                std::mt19937 rng(1);
                const std::vector<int> out = generateDevice(GemmDevicePath::Custom, weights, workspace, model, context, new_tokens, temperature, rng);
                sink += static_cast<float>(out.back());
            });
            freeDeviceWorkspace(workspace);
            freeDeviceWeights(weights);
            generation.gpu = true;
            generation.cublas_per_s = tokensPerSecond(cublas_ms, new_tokens);
            generation.custom_per_s = tokensPerSecond(custom_ms, new_tokens);
        }
        generated.push_back(generation);

        Tensor::seed(1);
        MultiHeadAttention attention_layer(4, 1, block, width);
        Tensor attention_input({1, block, width}, 0.0f);
        Tensor attention_grad({1, block, width}, 0.0f);
        for (size_t i = 0; i < attention_input.size(); ++i) {
            const float value = static_cast<float>((i * 17) % 100) / 50.0f - 1.0f;
            attention_input[i] = value;
            attention_grad[i] = value;
        }
        attention_layer.forward(attention_input);

        std::cout << "Timing attention backward " << shape << std::endl;
        Tensor::setAllowCudaGemm(false);
        Tensor::setAllowCustomGemm(false);
        AttentionRow att;
        att.shape = shape;
        att.cpu_ms = medianMs([&] {
            const Tensor cpu_grad = attention_layer.backward(attention_grad);
            sink += cpu_grad[0];
        });
        if (gpu_device) {
            DeviceBackwardWs attention_ws;
            if (!allocDeviceBackwardWs(width, 4, block, attention_ws)) {
                throw std::runtime_error("attention workspace allocation failed");
            }
            struct FreeWs {
                DeviceBackwardWs& ws;
                ~FreeWs() { freeDeviceBackwardWs(ws); }
            } free_ws{attention_ws};
            auto timePath = [&](GemmDevicePath path, double& milliseconds, float& diff) {
                AttentionState state;
                if (!forwardAttentionDevice(path, attention_layer, attention_input.getData().data(), block, state)) {
                    freeAttentionState(state);
                    throw std::runtime_error("attention forward failed");
                }
                AttentionGrad device_grad;
                bool ran = true;
                milliseconds = medianMs([&] {
                    if (!ran) {
                        return;
                    }
                    if (!backwardAttentionDevice(path, state, attention_layer, attention_ws, attention_grad.getData().data(), device_grad)) {
                        ran = false;
                        return;
                    }
                    sink += device_grad.input.empty() ? 0.0f : device_grad.input[0];
                });
                const Tensor cpu_grad = attention_layer.backward(attention_grad);
                diff = ran ? attentionDiff(attention_layer, cpu_grad, device_grad) : INFINITY;
                freeAttentionState(state);
                if (!ran) {
                    throw std::runtime_error("attention backward failed");
                }
                std::cout << "Attention diff " << (path == GemmDevicePath::Cublas ? "cublas" : "custom")
                    << " " << shape << " " << formatDiff(diff) << std::endl;
                if (diff > 1e-3f) {
                    ++attention_failures;
                }
            };
            timePath(GemmDevicePath::Cublas, att.cublas_ms, att.cublas_diff);
            timePath(GemmDevicePath::Custom, att.custom_ms, att.custom_diff);
            att.gpu = true;
        }
        attention.push_back(att);
    }

    omp_set_num_threads(default_threads);
    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);

    std::vector<HybridRow> hybrid;
    for (size_t width : {size_t{32}, size_t{128}}) {
        HybridRow row;
        row.shape = "width " + std::to_string(width) + ", layers 3, block 64, vocab 5256";
        if (gpu_device) {
            std::cout << "Timing hybrid step " << row.shape << std::endl;
            row.cublas_ms = hybridStepMs(width, GemmDevicePath::Cublas);
            row.custom_ms = hybridStepMs(width, GemmDevicePath::Custom);
            row.gpu = true;
        }
        hybrid.push_back(row);
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);

    std::vector<HybridRow> feed;
    for (size_t width : {size_t{32}, size_t{128}}) {
        HybridRow row;
        row.shape = "width " + std::to_string(width) + ", layers 3, block 64, vocab 5256";
        if (gpu_device) {
            std::cout << "Timing attention and feed-forward step " << row.shape << std::endl;
            row.cublas_ms = blockStepMs(width, GemmDevicePath::Cublas);
            row.custom_ms = blockStepMs(width, GemmDevicePath::Custom);
            row.gpu = true;
        }
        feed.push_back(row);
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);

    const std::string text = markdown(cpuBrand(), processors, default_threads, gpu_line, rows, generated, attention, hybrid, feed);
    std::filesystem::create_directories("bench");
    std::ofstream file("bench/RESULTS.md");
    if (!file) {
        throw std::runtime_error("bench/RESULTS.md could not be written");
    }
    file << text;
    std::cout << text;
    std::cerr << "sink " << sink << std::endl;
    return attention_failures == 0 ? 0 : 1;
}
