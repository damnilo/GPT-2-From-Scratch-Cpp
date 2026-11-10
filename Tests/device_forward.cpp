#include "../NumCPP/include/CudaGemm.h"
#include "../NumCPP/include/DeviceGemm.h"
#include "../NumCPP/include/Tensor.h"
#include "../Cuda/include/Generate.h"
#include "../Transformer/include/Model.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kAtol = 1e-3f;
constexpr size_t kVocab = 5256;
constexpr size_t kBlock = 64;
constexpr size_t kWidth = 32;
constexpr size_t kHeads = 4;
constexpr size_t kLayers = 3;

float maxAbsDiff(const Tensor& expected, const std::vector<float>& actual, size_t offset) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(actual[i] - expected[offset + i]));
    }
    return max_diff;
}

void report(const std::string& name, bool ok, float max_diff, int& failures) {
    std::cout << name << (ok ? " correct" : " failed");
    if (!ok) {
        std::cout << " (max abs diff " << max_diff << ")";
        ++failures;
    }
    std::cout << std::endl;
}

bool checkWindow(const Model& model, const DeviceWeights& weights, DeviceWorkspace& workspace, size_t length, GemmDevicePath path, const char* path_name, int& failures) {
    std::vector<int> ids(length);
    Tensor input({1, length}, 0.0f);
    for (size_t i = 0; i < length; ++i) {
        ids[i] = static_cast<int>(i % 50);
        input[i] = static_cast<float>(ids[i]);
    }

    const Tensor cpu = model.forward(input);
    const size_t offset = (length - 1) * kVocab;
    std::vector<float> probs(kVocab);
    const bool ran = cpu.size() == length * kVocab && forwardDevice(path, weights, workspace, ids.data(), length, probs.data());
    const float max_diff = ran ? maxAbsDiff(cpu, probs, offset) : 0.0f;
    const std::string name = std::string(path_name) + " T=" + std::to_string(length);
    report(name, ran && max_diff <= kAtol, max_diff, failures);
    return ran && max_diff <= kAtol;
}

}

int main() {
    if (!cudaGemmAvailable()) {
        std::cout << "GPU is not visible. device forward is skipped." << std::endl;
        return 0;
    }

    Tensor::seed(1);
    Tensor::setAllowCudaGemm(false);
    Tensor::setAllowCustomGemm(false);

    Model model(kVocab, kBlock, kWidth, kHeads, kLayers, 1, kBlock);
    DeviceWeights weights;
    DeviceWorkspace workspace;
    const bool ready = uploadDeviceWeights(model, weights) && allocDeviceWorkspace(model.getConfig(), workspace);
    if (!ready) {
        std::cout << "upload or allocation failed" << std::endl;
        freeDeviceWorkspace(workspace);
        freeDeviceWeights(weights);
        Tensor::setAllowCudaGemm(true);
        return 1;
    }

    int failures = 0;
    for (size_t length : {size_t{8}, kBlock}) {
        checkWindow(model, weights, workspace, length, GemmDevicePath::Cublas, "cublas", failures);
        checkWindow(model, weights, workspace, length, GemmDevicePath::Custom, "custom", failures);
    }

    freeDeviceWorkspace(workspace);
    freeDeviceWeights(weights);
    Tensor::setAllowCudaGemm(true);
    return failures == 0 ? 0 : 1;
}
