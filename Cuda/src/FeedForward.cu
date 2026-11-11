
#include "../include/FeedForward.h"
#include "../include/Kernels.cuh"
#include "../include/Linear.h"

bool forwardFeedForwardDevice(
    GemmDevicePath path,
    FeedForward& layer,
    const float* host_in,
    size_t T,
    FeedForwardState& state,
    std::vector<float>& host_out
) {
    if (host_in == nullptr || T == 0 || T > 1024) {
        return false;
    }
    if (!state.owns_weights && !uploadFeedForwardWeights(layer, state)) {
        return false;
    }

    const size_t C = state.C;
    const size_t F = state.F;
    if (C == 0 || F == 0 || F != 4 * C || F > 1024
        || state.W1 == nullptr || state.b1 == nullptr
        || state.W2 == nullptr || state.b2 == nullptr) {
        return false;
    }
    if (!prepareFeedForwardActivations(state, T, C)) {
        return false;
    }

    if (cudaMemcpy(state.x, host_in, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (!linearForwardDevice(path, state.x, state.W1, state.b1, state.hidden, T, C, F)) {
        return false;
    }
    cudaGELU<<<static_cast<unsigned>(T), static_cast<unsigned>(F)>>>(state.hidden, state.activated, T, F);
    if (!linearForwardDevice(path, state.activated, state.W2, state.b2, state.scratch, T, F, C)) {
        return false;
    }

    host_out.resize(T * C);
    if (cudaMemcpy(host_out.data(), state.scratch, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return true;
}

bool backwardFeedForwardCore(
    GemmDevicePath path,
    const FeedForwardState& state,
    FeedForward& layer,
    const float* grad_ptr,
    cudaMemcpyKind kind,
    float* device_dx,
    FeedForwardGrad& out
) {
    const size_t T = state.T;
    const size_t C = state.C;
    const size_t F = state.F;
    if (grad_ptr == nullptr || T == 0 || T > 1024 || C == 0 || F == 0 || F != 4 * C || F > 1024) {
        return false;
    }
    if (state.x == nullptr || state.hidden == nullptr || state.activated == nullptr
        || state.backward == nullptr
        || state.W1 == nullptr || state.W2 == nullptr
        || layer.parameters().size() != 4) {
        return false;
    }

    const size_t grad_n = T * C;
    const size_t act_n = T * F;
    float* grad = state.backward;
    float* d_act = grad + grad_n;
    float* scratch = d_act + act_n;

    if (cudaMemcpy(grad, grad_ptr, grad_n * sizeof(float), kind) != cudaSuccess) {
        return false;
    }

    out.W2.resize(C * F);
    out.b2.resize(C);
    if (!linearBackwardDevice(path, state.activated, state.W2, grad, d_act, out.W2.data(), out.b2.data(), scratch, T, F, C)) {
        return false;
    }

    cudaGELUBackward<<<static_cast<unsigned>((act_n + 255) / 256), 256>>>(state.hidden, d_act, act_n);

    out.W1.resize(F * C);
    out.b1.resize(F);
    if (!linearBackwardDevice(path, state.x, state.W1, d_act, grad, out.W1.data(), out.b1.data(), scratch, T, C, F)) {
        return false;
    }
    out.input.resize(grad_n);
    if (cudaMemcpy(out.input.data(), grad, grad_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (device_dx != nullptr
        && cudaMemcpy(device_dx, grad, grad_n * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }

    return true;
}

bool backwardFeedForwardDevice(
    GemmDevicePath path,
    const FeedForwardState& state,
    FeedForward& layer,
    const float* host_grad,
    FeedForwardGrad& out
) {
    return backwardFeedForwardCore(path, state, layer, host_grad, cudaMemcpyHostToDevice, nullptr, out);
}

bool forwardFeedForwardOnDevice(
    GemmDevicePath path,
    FeedForward& layer,
    const float* device_in,
    float* device_out,
    size_t T,
    FeedForwardState& state
) {
    if (device_in == nullptr || device_out == nullptr || T == 0 || T > 1024) {
        return false;
    }
    if (!state.owns_weights && !uploadFeedForwardWeights(layer, state)) {
        return false;
    }
    const size_t C = state.C;
    const size_t F = state.F;
    if (C == 0 || F == 0 || F != 4 * C || F > 1024
        || state.W1 == nullptr || state.b1 == nullptr
        || state.W2 == nullptr || state.b2 == nullptr) {
        return false;
    }
    if (!prepareFeedForwardActivations(state, T, C)) {
        return false;
    }
    if (device_in != state.x
        && cudaMemcpy(state.x, device_in, T * C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }
    if (!linearForwardDevice(path, state.x, state.W1, state.b1, state.hidden, T, C, F)) {
        return false;
    }
    cudaGELU<<<static_cast<unsigned>(T), static_cast<unsigned>(F)>>>(state.hidden, state.activated, T, F);
    if (!linearForwardDevice(path, state.activated, state.W2, state.b2, state.scratch, T, F, C)) {
        return false;
    }
    if (device_out == state.scratch) {
        return true;
    }
    return cudaMemcpy(device_out, state.scratch, T * C * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
}

bool backwardFeedForwardOnDevice(
    GemmDevicePath path,
    const FeedForwardState& state,
    FeedForward& layer,
    const float* device_grad,
    float* device_dx,
    FeedForwardGrad& out
) {
    return device_dx != nullptr
        && backwardFeedForwardCore(path, state, layer, device_grad, cudaMemcpyDeviceToDevice, device_dx, out);
}

bool refreshFeedForwardWeights(Model& model, std::vector<FeedForwardDevice>& slots) {
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!uploadFeedForwardWeights(model.block(i).feedForward(), slots[i].state)) {
            return false;
        }
    }
    return true;
}

bool enableDeviceFeedForward(Model& model, GemmDevicePath path, std::vector<FeedForwardDevice>& slots) {
    const ModelConfig config = model.getConfig();
    if (config.batch_size != 1) {
        return false;
    }

    slots.resize(config.n_layers);
    for (size_t i = 0; i < config.n_layers; ++i) {
        slots[i].path = path;
        if (!uploadFeedForwardWeights(model.block(i).feedForward(), slots[i].state)) {
            disableDeviceFeedForward(model, slots);
            return false;
        }
        model.block(i).feedForward().useDevice(&slots[i]);
    }
    return true;
}

void disableDeviceFeedForward(Model& model, std::vector<FeedForwardDevice>& slots) {
    for (size_t i = 0; i < slots.size(); ++i) {
        model.block(i).feedForward().useDevice(nullptr);
        freeFeedForwardState(slots[i].state);
    }
    slots.clear();
}