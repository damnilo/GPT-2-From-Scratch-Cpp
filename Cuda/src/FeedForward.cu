
#include "../include/FeedForward.h"
#include "../include/Kernels.cuh"

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
    if (!gemmDevice(path, state.x, state.W1, state.hidden, T, C, F)) {
        return false;
    }

    const GpuTensor b1{state.b1, F};
    cudaAddBias<<<dim3(static_cast<unsigned>((F + 255) / 256), static_cast<unsigned>(T)), 256>>>(state.hidden, b1, T, F);
    cudaGELU<<<static_cast<unsigned>(T), static_cast<unsigned>(F)>>>(state.hidden, state.activated, T, F);

    if (!gemmDevice(path, state.activated, state.W2, state.scratch, T, F, C)) {
        return false;
    }
    const GpuTensor b2{state.b2, C};
    cudaAddBias<<<dim3(static_cast<unsigned>((C + 255) / 256), static_cast<unsigned>(T)), 256>>>(state.scratch, b2, T, C);

    host_out.resize(T * C);
    if (cudaMemcpy(host_out.data(), state.scratch, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
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
    const size_t T = state.T;
    const size_t C = state.C;
    const size_t F = state.F;
    if (host_grad == nullptr || T == 0 || T > 1024 || C == 0 || F == 0 || F != 4 * C || F > 1024) {
        return false;
    }
    if (state.x == nullptr || state.hidden == nullptr || state.activated == nullptr
        || state.W1 == nullptr || state.W2 == nullptr
        || layer.parameters().size() != 4) {
        return false;
    }

    const size_t wider = T > C ? T : C;
    const size_t grad_n = T * C;
    const size_t trans_n = F * wider;
    const size_t dw_n = F * C;
    const size_t act_n = T * F;
    float* raw = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&raw), (grad_n + trans_n + dw_n + act_n + F) * sizeof(float)) != cudaSuccess) {
        return false;
    }

    float* grad = raw;
    float* trans = grad + grad_n;
    float* dw = trans + trans_n;
    float* d_act = dw + dw_n;
    float* db = d_act + act_n;
    auto fail = [&]() {
        cudaFree(raw);
        return false;
    };

    if (cudaMemcpy(grad, host_grad, grad_n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail();
    }

    cudaTranspose<<<static_cast<unsigned>(F), static_cast<unsigned>(C)>>>(state.W2, trans, F, C);
    if (!gemmDevice(path, grad, trans, d_act, T, C, F)) return fail();

    cudaTranspose<<<static_cast<unsigned>(T), static_cast<unsigned>(F)>>>(state.activated, trans, T, F);
    if (!gemmDevice(path, trans, grad, dw, F, T, C)) return fail();
    cudaTranspose<<<static_cast<unsigned>(F), static_cast<unsigned>(C)>>>(dw, trans, F, C);
    out.W2.resize(C * F);
    if (cudaMemcpy(out.W2.data(), trans, C * F * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaSumRows<<<1, static_cast<unsigned>(C)>>>(grad, db, 1, T, C);
    out.b2.resize(C);
    if (cudaMemcpy(out.b2.data(), db, C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaGELUBackward<<<static_cast<unsigned>((act_n + 255) / 256), 256>>>(state.hidden, d_act, act_n);

    cudaTranspose<<<static_cast<unsigned>(C), static_cast<unsigned>(F)>>>(state.W1, trans, C, F);
    if (!gemmDevice(path, d_act, trans, grad, T, F, C)) return fail();
    out.input.resize(grad_n);
    if (cudaMemcpy(out.input.data(), grad, grad_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaTranspose<<<static_cast<unsigned>(T), static_cast<unsigned>(C)>>>(state.x, trans, T, C);
    if (!gemmDevice(path, trans, d_act, dw, C, T, F)) return fail();
    cudaTranspose<<<static_cast<unsigned>(C), static_cast<unsigned>(F)>>>(dw, trans, C, F);
    out.W1.resize(F * C);
    if (cudaMemcpy(out.W1.data(), trans, F * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaSumRows<<<1, static_cast<unsigned>(F)>>>(d_act, db, 1, T, F);
    out.b1.resize(F);
    if (cudaMemcpy(out.b1.data(), db, F * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaFree(raw);
    return true;
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