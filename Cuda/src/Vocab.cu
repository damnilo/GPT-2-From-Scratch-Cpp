#include "../include/Vocab.h"
#include "../include/Linear.h"

#include <cuda_runtime.h>

bool forwardVocabDevice(
    GemmDevicePath path,
    Linear& layer,
    const float* host_in,
    size_t T,
    VocabState& state,
    std::vector<float>& host_logits
) {
    if (host_in == nullptr || T == 0 || T > 1024) {
        return false;
    }
    if (!state.owns_weights && !uploadVocabWeights(layer, state)) {
        return false;
    }

    const size_t C = state.C;
    const size_t V = state.V;
    if (C == 0 || V == 0 || C > 1024 || state.W == nullptr || state.b == nullptr) {
        return false;
    }
    if (!prepareVocabActivation(state, T, C, V)) {
        return false;
    }

    if (cudaMemcpy(state.x, host_in, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (!linearForwardDevice(path, state.x, state.W, state.b, state.logits, T, C, V)) {
        return false;
    }

    host_logits.resize(T * V);
    if (cudaMemcpy(host_logits.data(), state.logits, T * V * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return true;
}

bool backwardVocabDevice(
    GemmDevicePath path,
    const VocabState& state,
    Linear& layer,
    const float* host_grad,
    VocabGrad& out,
    float* device_dx
) {
    const size_t T = state.T;
    const size_t C = state.C;
    const size_t V = state.V;
    if (host_grad == nullptr || T == 0 || T > 1024 || C == 0 || C > 1024 || V == 0) {
        return false;
    }
    if (state.x == nullptr || state.W == nullptr || state.backward == nullptr || layer.parameters().size() != 2) {
        return false;
    }

    const size_t grad_n = T * V;
    const size_t dx_n = T * C;
    float* grad = state.backward;
    float* dx = grad + grad_n;
    float* scratch = dx + dx_n;

    if (cudaMemcpy(grad, host_grad, grad_n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    out.W.resize(C * V);
    out.b.resize(V);
    if (!linearBackwardDevice(path, state.x, state.W, grad, dx, out.W.data(), out.b.data(), scratch, T, C, V)) {
        return false;
    }
    out.input.resize(dx_n);
    if (cudaMemcpy(out.input.data(), dx, dx_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (device_dx != nullptr
        && cudaMemcpy(device_dx, dx, dx_n * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }

    return true;
}

bool forwardVocabOnDevice(
    GemmDevicePath path,
    Linear& layer,
    const float* device_in,
    size_t T,
    VocabState& state,
    std::vector<float>& host_logits
) {
    if (device_in == nullptr || T == 0 || T > 1024) {
        return false;
    }
    if (!state.owns_weights && !uploadVocabWeights(layer, state)) {
        return false;
    }
    const size_t C = state.C;
    const size_t V = state.V;
    if (C == 0 || V == 0 || C > 1024 || state.W == nullptr || state.b == nullptr) {
        return false;
    }
    if (!prepareVocabActivation(state, T, C, V)) {
        return false;
    }
    if (device_in != state.x
        && cudaMemcpy(state.x, device_in, T * C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }
    if (!linearForwardDevice(path, state.x, state.W, state.b, state.logits, T, C, V)) {
        return false;
    }
    host_logits.resize(T * V);
    return cudaMemcpy(host_logits.data(), state.logits, T * V * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool refreshVocabWeights(Model& model, VocabDevice& slot) {
    return uploadVocabWeights(model.vocabLinear(), slot.state);
}

bool enableDeviceVocab(Model& model, GemmDevicePath path, VocabDevice& slot) {
    if (model.getConfig().batch_size != 1) {
        return false;
    }
    slot.path = path;
    if (!uploadVocabWeights(model.vocabLinear(), slot.state)) {
        disableDeviceVocab(model, slot);
        return false;
    }
    model.vocabLinear().useDevice(&slot);
    return true;
}

void disableDeviceVocab(Model& model, VocabDevice& slot) {
    model.vocabLinear().useDevice(nullptr);
    freeVocabState(slot.state);
}
