#include "../Cuda/include/Block.h"
#include "../NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "../NeuralNet/Optimizers/include/AdamW.h"
#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/Tensor.h"
#include "../Transformer/include/Model.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kAtol = 1e-2f;
constexpr size_t kVocab = 5256;
constexpr size_t kLength = 64;
constexpr size_t kHeads = 4;
constexpr size_t kLayers = 3;

float compareParameters(const std::vector<Tensor*>& cpu, const std::vector<Tensor*>& device, size_t& worst) {
    if (cpu.size() != device.size()) {
        return INFINITY;
    }
    float diff = 0.0f;
    worst = 0;
    for (size_t tensor = 0; tensor < cpu.size(); ++tensor) {
        if (cpu[tensor]->size() != device[tensor]->size()) {
            worst = tensor;
            return INFINITY;
        }
        for (size_t i = 0; i < cpu[tensor]->size(); ++i) {
            const float gap = std::fabs((*cpu[tensor])[i] - (*device[tensor])[i]);
            if (gap > diff) {
                diff = gap;
                worst = tensor;
            }
        }
    }
    return diff;
}

void fillWindow(Tensor& input, Tensor& target) {
    for (size_t i = 0; i < kLength; ++i) {
        const size_t id = (i * 17) % kVocab;
        const size_t next = ((i + 1) * 17) % kVocab;
        input[i] = static_cast<float>(id);
        target[i * kVocab + next] = 1.0f;
    }
}

void report(const std::string& name, bool ok, float diff, size_t worst, int& failures) {
    std::cout << name << (ok ? " correct" : " failed");
    if (!ok) {
        std::cout << " (max abs diff " << diff << ", tensor " << worst << ")";
        ++failures;
    }
    std::cout << std::endl;
}

void checkCase(size_t width, GemmDevicePath path, const char* path_name, int& failures) {
    Tensor::seed(1);
    Model cpu(kVocab, kLength, width, kHeads, kLayers, 1, kLength);
    Tensor::seed(1);
    Model device(kVocab, kLength, width, kHeads, kLayers, 1, kLength);

    Tensor input({1, kLength}, 0.0f);
    Tensor target({1, kLength, kVocab}, 0.0f);
    fillWindow(input, target);

    AdamW cpu_optimizer(0.001f, 0.01f);
    cpu.zeroGrad();
    const Tensor cpu_out = cpu.forward(input);
    cpu.backward(CrossEntropyLoss::backward(cpu_out, target));
    cpu_optimizer.step(cpu.parameters(), cpu.gradients());

    const std::string name = std::string(path_name) + " width=" + std::to_string(width);
    StackDevice stack;
    if (!enableDeviceStack(device, path, stack)) {
        disableDeviceStack(device, stack);
        report(name, false, INFINITY, 0, failures);
        return;
    }

    AdamW device_optimizer(0.001f, 0.01f);
    if (!loadAdamState(device, stack, device_optimizer)) {
        disableDeviceStack(device, stack);
        report(name, false, INFINITY, 0, failures);
        return;
    }

    device.zeroGrad();
    device.forward(input);
    float loss = 0.0f;
    std::vector<int> classes(kLength);
    for (size_t i = 0; i < kLength; ++i) {
        classes[i] = static_cast<int>(((i + 1) * 17) % kVocab);
    }
    if (!lossStackDevice(stack, classes.data(), loss)) {
        disableDeviceStack(device, stack);
        report(name, false, INFINITY, 0, failures);
        return;
    }
    (void)loss;
    device.backward(Tensor());

    device_optimizer.step({}, {});
    if (!adamDevice(stack, device_optimizer) || !storeAdamState(device, stack, device_optimizer)) {
        disableDeviceStack(device, stack);
        report(name, false, INFINITY, 0, failures);
        return;
    }
    disableDeviceStack(device, stack);

    size_t worst = 0;
    const float diff = compareParameters(cpu.parameters(), device.parameters(), worst);
    report(name, diff <= kAtol, diff, worst, failures);
}

}

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. adam step is skipped." << std::endl;
        return 0;
    }

    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(false);

    int failures = 0;
    for (size_t width : {size_t{32}, size_t{128}}) {
        checkCase(width, GemmDevicePath::Cublas, "cublas", failures);
        checkCase(width, GemmDevicePath::Custom, "custom", failures);
    }

    Tensor::setAllowCudaGemm(true);
    Tensor::setAllowCustomGemm(false);
    return failures == 0 ? 0 : 1;
}
