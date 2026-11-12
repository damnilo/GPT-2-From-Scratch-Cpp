
#include "../include/Attention.h"
#include "../include/Kernels.cuh"
#include "../include/Linear.h"

#include <cmath>
#include <cstddef>

namespace {

bool attentionForwardKernels(GemmDevicePath path, AttentionState& state, size_t T, float* proj_out) {
    const size_t H = state.n_heads;
    const size_t C = state.C;
    const size_t d = state.d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    cudaGather3d<<<grid, static_cast<unsigned>(d)>>>(state.joined, state.x, H, T, C, d);
    if (!linearBatchForwardDevice(path, state.x, state.Wq, state.bq, state.q, H, T, d, d)) return false;
    if (!linearBatchForwardDevice(path, state.x, state.Wk, state.bk, state.k, H, T, d, d)) return false;
    if (!linearBatchForwardDevice(path, state.x, state.Wv, state.bv, state.v, H, T, d, d)) return false;

    cudaTranspose3d<<<grid, static_cast<unsigned>(d)>>>(state.k, state.kt, H, T, d);
    if (!gemmBatchDevice(path, state.q, state.kt, state.scores, H, T, d, T)) return false;
    cudaMaskSoftmax3d<<<grid, 1>>>(state.scores, H, T, scale);
    if (!gemmBatchDevice(path, state.scores, state.v, state.ctx, H, T, T, d)) return false;
    if (!linearBatchForwardDevice(path, state.ctx, state.Wo_h, state.bo_h, state.head_out, H, T, d, d)) return false;
    cudaScatter3d<<<grid, static_cast<unsigned>(d)>>>(state.head_out, state.joined, H, T, C, d);
    return linearForwardDevice(path, state.joined, state.Wo, state.bo, proj_out, T, C, C);
}

}

bool forwardAttentionDevice(
    GemmDevicePath path,
    MultiHeadAttention& layer,
    const float* host_in,
    size_t T,
    AttentionState& state,
    std::vector<float>* host_out
) {
    if (host_in == nullptr || T == 0 || T > 1024) {
        return false;
    }
    if (!state.owns_weights && !uploadAttentionWeights(layer, state)) {
        return false;
    }

    const size_t H = state.n_heads;
    const size_t C = state.C;
    const size_t d = state.d;
    if (H == 0 || C == 0 || d == 0 || C != H * d
        || state.Wq == nullptr || state.Wk == nullptr || state.Wv == nullptr || state.Wo_h == nullptr
        || state.bq == nullptr || state.bk == nullptr || state.bv == nullptr || state.bo_h == nullptr
        || state.Wo == nullptr || state.bo == nullptr) {
        return false;
    }
    if (!prepareAttentionActivations(state, H, T, C, d)) {
        return false;
    }

    if (state.kt == nullptr || state.head_out == nullptr) {
        return false;
    }

    if (cudaMemcpy(state.joined, host_in, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (host_out == nullptr) {
        return attentionForwardKernels(path, state, T, state.kt);
    }
    if (!attentionForwardKernels(path, state, T, state.kt)) {
        return false;
    }
    host_out->resize(T * C);
    return cudaMemcpy(host_out->data(), state.kt, T * C * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool backwardAttentionCore(
    GemmDevicePath path,
    const AttentionState& state,
    MultiHeadAttention& layer,
    DeviceBackwardWs& ws,
    const float* grad_ptr,
    cudaMemcpyKind kind,
    float* device_dx,
    AttentionGrad* out
) {

    size_t T = state.T;
    size_t H = state.n_heads;
    size_t d = state.d;
    size_t C = state.C;

    if (T == 0 
        || T > ws.capacity
        || H != ws.n_heads
        || d != ws.d
        || C != ws.C 
        || grad_ptr == nullptr
        || state.scratch == nullptr
        || layer.parameters().size() != ws.n_heads * 8 + 2
    ) {
        return false;
    }

    float scale = 1.0f / sqrtf(d);
    float* scratch = state.scratch;

    if (cudaMemcpy(ws.grad, grad_ptr, T * C * sizeof(float), kind) != cudaSuccess) {
        return false;
    }

    float* host_wo = nullptr;
    float* host_bo = nullptr;
    if (out != nullptr) {
        out->Wo.resize(C * C);
        out->bo.resize(C);
        host_wo = out->Wo.data();
        host_bo = out->bo.data();
    }
    if (!linearBackwardDevice(path, state.joined, state.Wo, ws.grad, ws.d_joined, host_wo, host_bo, scratch, T, C, C, state.dWo, state.dbo)) {
        return false;
    }

    cudaGather3d<<<dim3(T, H), d>>>(ws.d_joined, ws.ctx, H, T, C, d);

    std::vector<float> packed_w;
    std::vector<float> packed_b;
    float* host_w = nullptr;
    float* host_b = nullptr;
    if (out != nullptr) {
        packed_w.resize(H * d * d);
        packed_b.resize(H * d);
        host_w = packed_w.data();
        host_b = packed_b.data();
        out->heads.resize(H);
    }
    auto saveHead = [&](size_t h, std::vector<float>& weights, std::vector<float>& bias) {
        const auto w0 = static_cast<std::ptrdiff_t>(h * d * d);
        const auto b0 = static_cast<std::ptrdiff_t>(h * d);
        weights.assign(packed_w.begin() + w0, packed_w.begin() + w0 + static_cast<std::ptrdiff_t>(d * d));
        bias.assign(packed_b.begin() + b0, packed_b.begin() + b0 + static_cast<std::ptrdiff_t>(d));
    };

    if (!linearBatchBackwardDevice(path, state.ctx, state.Wo_h, ws.ctx, ws.d_ctx, host_w, host_b, scratch, H, T, d, d, state.dWo_h, state.dbo_h)) {
        return false;
    }
    if (out != nullptr) {
        for (size_t h = 0; h < H; ++h) {
            saveHead(h, out->heads[h].Wo, out->heads[h].bo);
        }
    }

    cudaTranspose3d<<<dim3(T, H), T>>>(state.scores, ws.scores, H, T, T);
    if (!gemmBatchDevice(path, ws.scores, ws.d_ctx, ws.d_v, H, T, T, d)) return false;

    cudaTranspose3d<<<dim3(T, H), d>>>(state.v, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.d_ctx, ws.kt, ws.d_p, H, T, d, T)) return false;

    cudaSoftmaxBackward3d<<<dim3(T, H), 1>>>(state.scores, ws.d_p, ws.d_scores, H, T, scale);
    if (!gemmBatchDevice(path, ws.d_scores, state.k, ws.d_q, H, T, T, d)) return false;

    cudaTranspose3d<<<dim3(T, H), T>>>(ws.d_scores, ws.scores, H, T, T);
    if (!gemmBatchDevice(path, ws.scores, state.q, ws.d_k, H, T, T, d)) return false;

    const size_t n = H * T * d;

    if (!linearBatchBackwardDevice(path, state.x, state.Wq, ws.d_q, ws.d_x, host_w, host_b, scratch, H, T, d, d, state.dWq, state.dbq)) {
        return false;
    }
    if (out != nullptr) {
        for (size_t h = 0; h < H; ++h) {
            saveHead(h, out->heads[h].Wq, out->heads[h].bq);
        }
    }

    if (!linearBatchBackwardDevice(path, state.x, state.Wk, ws.d_k, ws.q, host_w, host_b, scratch, H, T, d, d, state.dWk, state.dbk)) {
        return false;
    }
    if (out != nullptr) {
        for (size_t h = 0; h < H; ++h) {
            saveHead(h, out->heads[h].Wk, out->heads[h].bk);
        }
    }

    cudaAddVectors<<<(n + 255) / 256, 256>>>(ws.d_x, ws.q, n);

    if (!linearBatchBackwardDevice(path, state.x, state.Wv, ws.d_v, ws.q, host_w, host_b, scratch, H, T, d, d, state.dWv, state.dbv)) {
        return false;
    }
    if (out != nullptr) {
        for (size_t h = 0; h < H; ++h) {
            saveHead(h, out->heads[h].Wv, out->heads[h].bv);
        }
    }

    cudaAddVectors<<<(n + 255) / 256, 256>>>(ws.d_x, ws.q, n);

    cudaScatter3d<<<dim3(T, H), d>>>(ws.d_x, ws.d_input, H, T, C, d);
    if (device_dx != nullptr
        && cudaMemcpy(device_dx, ws.d_input, T * C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }
    if (out != nullptr) {
        out->input.resize(T * C);
        if (cudaMemcpy(out->input.data(), ws.d_input, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
    }

    return true;
}

bool backwardAttentionDevice(
    GemmDevicePath path,
    const AttentionState& state,
    MultiHeadAttention& layer,
    DeviceBackwardWs& ws,
    const float* host_grad,
    AttentionGrad& out
) {
    return backwardAttentionCore(path, state, layer, ws, host_grad, cudaMemcpyHostToDevice, nullptr, &out);
}

bool forwardAttentionOnDevice(
    GemmDevicePath path,
    MultiHeadAttention& layer,
    const float* device_in,
    float* device_out,
    size_t T,
    AttentionState& state
) {
    if (device_in == nullptr || device_out == nullptr || T == 0 || T > 1024) {
        return false;
    }
    if (!state.owns_weights && !uploadAttentionWeights(layer, state)) {
        return false;
    }
    const size_t H = state.n_heads;
    const size_t C = state.C;
    const size_t d = state.d;
    if (H == 0 || C == 0 || d == 0 || C != H * d || state.Wo == nullptr || state.bo == nullptr) {
        return false;
    }
    if (!prepareAttentionActivations(state, H, T, C, d) || state.kt == nullptr || state.head_out == nullptr) {
        return false;
    }
    if (device_in != state.joined
        && cudaMemcpy(state.joined, device_in, T * C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }
    return attentionForwardKernels(path, state, T, device_out);
}

bool backwardAttentionOnDevice(
    GemmDevicePath path,
    const AttentionState& state,
    MultiHeadAttention& layer,
    DeviceBackwardWs& ws,
    const float* device_grad,
    float* device_dx,
    AttentionGrad* out
) {
    return device_dx != nullptr
        && backwardAttentionCore(path, state, layer, ws, device_grad, cudaMemcpyDeviceToDevice, device_dx, out);
}

bool enableDeviceAttention(
    Model& model,
    GemmDevicePath path,
    std::vector<AttentionDevice>& slots
) {
    ModelConfig config = model.getConfig();
    if (config.batch_size != 1) return false;

    slots.resize(config.n_layers);
    
    for (int i = 0; i < config.n_layers; ++i) {
        slots[i].path = path;
        if (!allocDeviceBackwardWs(config.embedding_dim, config.n_heads, config.block_size, slots[i].ws)
            || !uploadAttentionWeights(model.block(i).attention(), slots[i].state)) {
            disableDeviceAttention(model, slots);
            return false;
        }

        model.block(i).attention().useDevice(&slots[i]);
    }

    return true;
}

bool refreshAttentionWeights(Model& model, std::vector<AttentionDevice>& slots) {
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!uploadAttentionWeights(model.block(i).attention(), slots[i].state)) {
            return false;
        }
    }
    return true;
}

void disableDeviceAttention(
    Model& model,
    std::vector<AttentionDevice>& slots
) {
    for (int i = 0; i < slots.size(); ++i) {
        model.block(i).attention().useDevice(nullptr);
        freeAttentionState(slots[i].state);
        freeDeviceBackwardWs(slots[i].ws);
    }

    slots.clear();
}