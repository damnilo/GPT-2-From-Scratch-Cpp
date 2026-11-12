#include "../Cuda/include/Vocab.h"
#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/Tensor.h"
#include "../NeuralNet/Layers/include/Linear.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kAtol = 1e-3f;
constexpr size_t kVocab = 5256;

float maxAbs(const std::vector<float>& got, const Tensor& expected) {
    if (got.size() != expected.size()) {
        return INFINITY;
    }
    float diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::fabs(got[i] - expected[i]));
    }
    return diff;
}

void fillSignal(Tensor& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<float>((i * 17) % 100) / 50.0f - 1.0f;
    }
}

void report(const std::string& name, bool ok, float diff, int& failures) {
    std::cout << name << (ok ? " correct" : " failed");
    if (!ok) {
        std::cout << " (max abs diff " << diff << ")";
        ++failures;
    }
    std::cout << std::endl;
}

void checkCase(Linear& layer, size_t width, size_t length, GemmDevicePath path, const char* path_name, int& failures) {
    Tensor input({1, length, width}, 0.0f);
    Tensor grad({1, length, kVocab}, 0.0f);
    fillSignal(input);
    fillSignal(grad);

    layer.zeroGrad();
    const Tensor cpu_out = layer.forward(input);
    const Tensor cpu_in = layer.backward(grad);

    VocabState state;
    VocabGrad device;
    std::vector<float> host_out;
    const bool ran = forwardVocabDevice(path, layer, input.getData().data(), length, state, host_out)
        && backwardVocabDevice(path, state, layer, grad.getData().data(), &device);
    float diff = INFINITY;
    if (ran) {
        const std::vector<Tensor*> grads = layer.gradients();
        diff = maxAbs(host_out, cpu_out);
        diff = std::max(diff, maxAbs(device.input, cpu_in));
        if (grads.size() != 2) {
            diff = INFINITY;
        } else {
            diff = std::max(diff, maxAbs(device.W, *grads[0]));
            diff = std::max(diff, maxAbs(device.b, *grads[1]));
        }
    }
    const std::string name = std::string(path_name) + " width=" + std::to_string(width) + " T=" + std::to_string(length);
    report(name, ran && diff <= kAtol, diff, failures);
    freeVocabState(state);
}

}

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. vocabulary backward is skipped." << std::endl;
        return 0;
    }

    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(false);

    int failures = 0;
    for (size_t width : {size_t{32}, size_t{128}}) {
        Tensor::seed(1);
        Linear layer(width, kVocab);
        for (size_t length : {size_t{8}, size_t{64}}) {
            checkCase(layer, width, length, GemmDevicePath::Cublas, "cublas", failures);
            checkCase(layer, width, length, GemmDevicePath::Custom, "custom", failures);
        }
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);
    return failures == 0 ? 0 : 1;
}
