#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/DeviceGemm.h"
#include "../NumCPP/include/Tensor.h"
#include "../Cuda/include/Attention.h"
#include "../Transformer/include/MultiHeadAttention.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kAtol = 1e-3f;
constexpr size_t kHeads = 4;
constexpr size_t kCapacity = 64;

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

float compareGradients(MultiHeadAttention& layer, const Tensor& input_grad, const AttentionGrad& got) {
    float diff = maxAbs(got.input, input_grad);
    const std::vector<Tensor*> grads = layer.gradients();
    if (got.heads.size() * 8 + 2 != grads.size()) {
        return INFINITY;
    }
    for (size_t h = 0; h < got.heads.size(); ++h) {
        const size_t base = h * 8;
        const HeadGrad& head = got.heads[h];
        diff = std::max(diff, maxAbs(head.Wq, *grads[base]));
        diff = std::max(diff, maxAbs(head.bq, *grads[base + 1]));
        diff = std::max(diff, maxAbs(head.Wk, *grads[base + 2]));
        diff = std::max(diff, maxAbs(head.bk, *grads[base + 3]));
        diff = std::max(diff, maxAbs(head.Wv, *grads[base + 4]));
        diff = std::max(diff, maxAbs(head.bv, *grads[base + 5]));
        diff = std::max(diff, maxAbs(head.Wo, *grads[base + 6]));
        diff = std::max(diff, maxAbs(head.bo, *grads[base + 7]));
    }
    diff = std::max(diff, maxAbs(got.Wo, *grads[grads.size() - 2]));
    diff = std::max(diff, maxAbs(got.bo, *grads[grads.size() - 1]));
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

void checkCase(MultiHeadAttention& layer, DeviceBackwardWs& workspace, size_t width, size_t length, GemmDevicePath path, const char* path_name, int& failures) {
    Tensor input({1, length, width}, 0.0f);
    Tensor grad({1, length, width}, 0.0f);
    fillSignal(input);
    fillSignal(grad);

    layer.zeroGrad();
    layer.forward(input);
    const Tensor cpu_input = layer.backward(grad);

    AttentionState state;
    AttentionGrad device;
    const bool ran = forwardAttentionDevice(path, layer, input.getData().data(), length, state)
        && backwardAttentionDevice(path, state, layer, workspace, grad.getData().data(), device);
    const float diff = ran ? compareGradients(layer, cpu_input, device) : INFINITY;
    const std::string name = std::string(path_name) + " width=" + std::to_string(width) + " T=" + std::to_string(length);
    report(name, ran && diff <= kAtol, diff, failures);

    freeAttentionState(state);
}

}

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. device backward is skipped." << std::endl;
        return 0;
    }

    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(false);

    int failures = 0;
    for (size_t width : {size_t{32}, size_t{128}}) {
        Tensor::seed(1);
        MultiHeadAttention layer(kHeads, 1, kCapacity, width);
        DeviceBackwardWs workspace;
        if (!allocDeviceBackwardWs(width, kHeads, kCapacity, workspace)) {
            std::cout << "allocation failed" << std::endl;
            freeDeviceBackwardWs(workspace);
            Tensor::setAllowCudaGemm(true);
            return 1;
        }

        for (size_t length : {size_t{8}, kCapacity}) {
            checkCase(layer, workspace, width, length, GemmDevicePath::Cublas, "cublas", failures);
            checkCase(layer, workspace, width, length, GemmDevicePath::Custom, "custom", failures);
        }
        freeDeviceBackwardWs(workspace);
    }

    Tensor::setAllowCudaGemm(true);
    return failures == 0 ? 0 : 1;
}
