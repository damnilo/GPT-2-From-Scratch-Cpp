#include "../Cuda/include/Vocab.h"
#include "../NeuralNet/Layers/include/Linear.h"
#include "../NeuralNet/Layers/include/Softmax.h"
#include "../NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/Tensor.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if GPT_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

constexpr float kAtol = 1e-3f;
constexpr size_t kVocab = 5256;

void report(const std::string& name, bool ok, float diff, int& failures) {
    std::cout << name << (ok ? " correct" : " failed");
    if (!ok) {
        std::cout << " (max abs diff " << diff << ")";
        ++failures;
    }
    std::cout << std::endl;
}

#if GPT_HAS_CUDA

float compare(const VocabState& state, const std::vector<float>& logits, const std::vector<int>& classes, float device_loss) {
    const size_t T = classes.size();
    if (T == 0 || logits.size() % T != 0) {
        return INFINITY;
    }
    const size_t V = logits.size() / T;
    Tensor values({1, T, V}, logits);
    Tensor target({1, T, V}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        target[t * V + static_cast<size_t>(classes[t])] = 1.0f;
    }

    Softmax softmax(-1);
    const Tensor probs = softmax.forward(values);
    const float cpu_loss = CrossEntropyLoss::loss(probs, target);
    const Tensor cpu_grad = softmax.backward(CrossEntropyLoss::backward(probs, target));

    std::vector<float> device_grad(T * V);
    if (cudaMemcpy(device_grad.data(), state.backward, device_grad.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess
        || cpu_grad.size() != device_grad.size()) {
        return INFINITY;
    }

    float diff = std::fabs(device_loss - cpu_loss);
    for (size_t i = 0; i < device_grad.size(); ++i) {
        diff = std::max(diff, std::fabs(device_grad[i] - cpu_grad[i]));
    }
    return diff;
}

void checkFloor(int& failures) {
    constexpr size_t T = 2;
    constexpr size_t V = 4;
    const std::vector<float> logits = {
        -100.0f, 0.0f, 0.0f, 0.0f,
        0.2f, -0.3f, 0.5f, 1.1f
    };
    const std::vector<int> classes = {0, 2};

    VocabState state;
    float device_loss = 0.0f;
    const bool ran = prepareVocabActivation(state, T, 1, V)
        && cudaMemcpy(state.logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess
        && softmaxCELossDevice(state, classes.data(), device_loss);
    const float diff = ran ? compare(state, logits, classes, device_loss) : INFINITY;
    report("floor", ran && diff <= kAtol, diff, failures);
    freeVocabState(state);
}

void fillSignal(Tensor& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<float>((i * 17) % 100) / 50.0f - 1.0f;
    }
}

void checkProjection(Linear& layer, size_t width, size_t length, GemmDevicePath path, const char* path_name, int& failures) {
    Tensor input({1, length, width}, 0.0f);
    fillSignal(input);

    std::vector<int> classes(length);
    for (size_t t = 0; t < length; ++t) {
        classes[t] = static_cast<int>(((t + 1) * 17) % kVocab);
    }

    VocabState state;
    std::vector<float> logits;
    float device_loss = 0.0f;
    const bool ran = forwardVocabDevice(path, layer, input.getData().data(), length, state, logits)
        && softmaxCELossDevice(state, classes.data(), device_loss);
    const float diff = ran ? compare(state, logits, classes, device_loss) : INFINITY;
    const std::string name = std::string(path_name) + " width=" + std::to_string(width) + " T=" + std::to_string(length);
    report(name, ran && diff <= kAtol, diff, failures);
    freeVocabState(state);
}

#endif

}

int main() {
#if !GPT_HAS_CUDA
    std::cout << "GPU is not visible. ce loss is skipped." << std::endl;
    return 0;
#else
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. ce loss is skipped." << std::endl;
        return 0;
    }

    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(false);

    int failures = 0;
    checkFloor(failures);
    for (size_t width : {size_t{32}, size_t{128}}) {
        Tensor::seed(1);
        Linear layer(width, kVocab);
        for (size_t length : {size_t{8}, size_t{64}}) {
            checkProjection(layer, width, length, GemmDevicePath::Cublas, "cublas", failures);
            checkProjection(layer, width, length, GemmDevicePath::Custom, "custom", failures);
        }
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);
    return failures == 0 ? 0 : 1;
#endif
}
