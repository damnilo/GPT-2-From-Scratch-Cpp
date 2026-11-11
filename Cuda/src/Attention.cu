
#include "../include/Attention.h"
#include "../include/Kernels.cuh"

#include <cmath>


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

    const size_t act_n = H * T * d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    float* kt = nullptr;
    float* head_out = nullptr;
    auto fail_local = [&]() {
        cudaFree(kt);
        cudaFree(head_out);
        return false;
    };
    if (cudaMalloc(reinterpret_cast<void**>(&kt), act_n * sizeof(float)) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&head_out), act_n * sizeof(float)) != cudaSuccess) {
        return fail_local();
    }

    if (cudaMemcpy(state.joined, host_in, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail_local();
    }

    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    cudaGather3d<<<grid, static_cast<unsigned>(d)>>>(state.joined, state.x, H, T, C, d);
    if (!gemmBatchDevice(path, state.x, state.Wq, state.q, H, T, d, d)) return fail_local();
    cudaAddBias3d<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(state.q, state.bq, H, T, d);
    if (!gemmBatchDevice(path, state.x, state.Wk, state.k, H, T, d, d)) return fail_local();
    cudaAddBias3d<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(state.k, state.bk, H, T, d);
    if (!gemmBatchDevice(path, state.x, state.Wv, state.v, H, T, d, d)) return fail_local();
    cudaAddBias3d<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(state.v, state.bv, H, T, d);

    cudaTranspose3d<<<grid, static_cast<unsigned>(d)>>>(state.k, kt, H, T, d);
    if (!gemmBatchDevice(path, state.q, kt, state.scores, H, T, d, T)) return fail_local();
    cudaMaskSoftmax3d<<<grid, 1>>>(state.scores, H, T, scale);
    if (!gemmBatchDevice(path, state.scores, state.v, state.ctx, H, T, T, d)) return fail_local();
    if (!gemmBatchDevice(path, state.ctx, state.Wo_h, head_out, H, T, d, d)) return fail_local();
    cudaAddBias3d<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(head_out, state.bo_h, H, T, d);
    cudaScatter3d<<<grid, static_cast<unsigned>(d)>>>(head_out, state.joined, H, T, C, d);

    if (host_out != nullptr) {
        if (!gemmDevice(path, state.joined, state.Wo, kt, T, C, C)) {
            return fail_local();
        }
        const GpuTensor bo{state.bo, C};
        cudaAddBias<<<dim3(static_cast<unsigned>((C + 255) / 256), static_cast<unsigned>(T)), 256>>>(kt, bo, T, C);
        host_out->resize(T * C);
        if (cudaMemcpy(host_out->data(), kt, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail_local();
        }
    }

    cudaFree(kt);
    cudaFree(head_out);
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

    size_t T = state.T;
    size_t H = state.n_heads;
    size_t d = state.d;
    size_t C = state.C;

    if (T == 0 
        || T > ws.capacity
        || H != ws.n_heads
        || d != ws.d
        || C != ws.C 
        || host_grad == nullptr
        || layer.parameters().size() != ws.n_heads * 8 + 2
    ) {
        return false;
    }

    float scale = 1.0f / sqrtf(d);
    float* w_t = NULL;
    float* dw = NULL;
    float* db = NULL;

    if (cudaMalloc(reinterpret_cast<void**>(&w_t), C * C * sizeof(float)) != cudaSuccess) return false;

    if (cudaMalloc(reinterpret_cast<void**>(&dw), C * C * sizeof(float)) != cudaSuccess) {
        cudaFree(w_t);
        return false;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&db), C * sizeof(float)) != cudaSuccess) {
        cudaFree(w_t);
        cudaFree(dw);
        return false;
    }

    auto fail = [&]() {
        cudaFree(w_t);
        cudaFree(dw);
        cudaFree(db);
        return false;
    };
    
    if (cudaMemcpy(ws.grad, host_grad, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail();
    }

    cudaTranspose3d<<<C, C>>>(state.Wo, w_t, 1, C, C);
    if (!gemmDevice(path, ws.grad, w_t, ws.d_joined, T, C, C)) return fail();

    cudaTranspose3d<<<T, C>>>(state.joined, ws.x, 1, T, C);
    if (!gemmDevice(path, ws.x, ws.grad, dw, C, T, C)) return fail();

    cudaTranspose3d<<<C, C>>>(dw, w_t, 1, C, C);
    out.Wo.resize(C * C);
    out.bo.resize(C);
    if (cudaMemcpy(out.Wo.data(), w_t, C * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }
    cudaSumRows<<<1, C>>>(ws.grad, db, 1, T, C);
    if (cudaMemcpy(out.bo.data(), db, C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaGather3d<<<dim3(T, H), d>>>(ws.d_joined, ws.ctx, H, T, C, d);

    cudaTranspose3d<<<dim3(d, H), d>>>(state.Wo_h, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.ctx, w_t, ws.d_ctx, H, T, d, d)) return fail();

    cudaTranspose3d<<<dim3(T, H), d>>>(state.ctx, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.ctx, dw, H, d, T, d)) return fail();

    out.heads.resize(H);
    for (size_t h = 0; h < H; ++h) {
        cudaTranspose3d<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
        out.heads[h].Wo.resize(d * d);
        if (cudaMemcpy(out.heads[h].Wo.data(), w_t, d * d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }

    cudaSumRows<<<H, d>>>(ws.ctx, db, H, T, d);
    for (size_t h = 0; h < H; ++h) {
        out.heads[h].bo.resize(d);
        if (cudaMemcpy(out.heads[h].bo.data(), db + h * d, d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }

    cudaTranspose3d<<<dim3(T, H), T>>>(state.scores, ws.scores, H, T, T);
    if (!gemmBatchDevice(path, ws.scores, ws.d_ctx, ws.d_v, H, T, T, d)) return fail();

    cudaTranspose3d<<<dim3(T, H), d>>>(state.v, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.d_ctx, ws.kt, ws.d_p, H, T, d, T)) return fail();

    cudaSoftmaxBackward3d<<<dim3(T, H), 1>>>(state.scores, ws.d_p, ws.d_scores, H, T, scale);
    if (!gemmBatchDevice(path, ws.d_scores, state.k, ws.d_q, H, T, T, d)) return fail();

    cudaTranspose3d<<<dim3(T, H), T>>>(ws.d_scores, ws.scores, H, T, T);
    if (!gemmBatchDevice(path, ws.scores, state.q, ws.d_k, H, T, T, d)) return fail();

    size_t n = H * T * d;

    cudaTranspose3d<<<dim3(d, H), d>>>(state.Wq, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.d_q, w_t, ws.d_x, H, T, d, d)) return fail();

    cudaTranspose3d<<<dim3(T, H), d>>>(state.x, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.d_q, dw, H, d, T, d)) return fail();

    for (size_t h = 0; h < H; ++h) {
        cudaTranspose3d<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
        out.heads[h].Wq.resize(d * d);
        if (cudaMemcpy(out.heads[h].Wq.data(), w_t, d * d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }

    cudaSumRows<<<H, d>>>(ws.d_q, db, H, T, d);
    for (size_t h = 0; h < H; ++h) {
        out.heads[h].bq.resize(d);
        if (cudaMemcpy(out.heads[h].bq.data(), db + h * d, d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }
    
    cudaTranspose3d<<<dim3(d, H), d>>>(state.Wk, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.d_k, w_t, ws.q, H, T, d, d)) return fail();

    cudaTranspose3d<<<dim3(T, H), d>>>(state.x, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.d_k, dw, H, d, T, d)) return fail();

    for (size_t h = 0; h < H; ++h) {
        cudaTranspose3d<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
        out.heads[h].Wk.resize(d * d);
        if (cudaMemcpy(out.heads[h].Wk.data(), w_t, d * d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }

    cudaSumRows<<<H, d>>>(ws.d_k, db, H, T, d);
    for (size_t h = 0; h < H; ++h) {
        out.heads[h].bk.resize(d);
        if (cudaMemcpy(out.heads[h].bk.data(), db + h * d, d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }
    
    cudaAddVectors<<<(n + 255) / 256, 256>>>(ws.d_x, ws.q, n);
    
    cudaTranspose3d<<<dim3(d, H), d>>>(state.Wv, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.d_v, w_t, ws.q, H, T, d, d)) return fail();

    cudaTranspose3d<<<dim3(T, H), d>>>(state.x, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.d_v, dw, H, d, T, d)) return fail();

    for (size_t h = 0; h < H; ++h) {
        cudaTranspose3d<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
        out.heads[h].Wv.resize(d * d);
        if (cudaMemcpy(out.heads[h].Wv.data(), w_t, d * d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }

    cudaSumRows<<<H, d>>>(ws.d_v, db, H, T, d);
    for (size_t h = 0; h < H; ++h) {
        out.heads[h].bv.resize(d);
        if (cudaMemcpy(out.heads[h].bv.data(), db + h * d, d * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return fail();
        }
    }

    cudaAddVectors<<<(n + 255) / 256, 256>>>(ws.d_x, ws.q, n);

    cudaScatter3d<<<dim3(T, H), d>>>(ws.d_x, ws.d_input, H, T, C, d);
    out.input.resize(T * C);
    if (cudaMemcpy(out.input.data(), ws.d_input, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaFree(w_t);
    cudaFree(dw);
    cudaFree(db);
    return true;
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