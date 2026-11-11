
#include "../include/Structs.h"

#include "../../Transformer/include/MultiHeadAttention.h"

#include <cuda_runtime.h>
#include <stdexcept>

struct Cursor {
    std::vector<Tensor*> params;
    size_t index = 0;

    const Tensor& next(const std::vector<size_t>& shape) {
        if (index >= params.size()) {
            throw std::runtime_error("not enough parameters");
        }

        const Tensor& tensor = *params[index];
        index += 1;

        if (tensor.getShape() != shape) {
            throw std::runtime_error("parameter shape mismatch");
        }

        return tensor;
    }
};

GpuTensor uploadPlain(const Tensor& tensor) {
    
    size_t count = tensor.size();
    float* ptr = nullptr;

    if (cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)) != cudaSuccess) return {};
    
    if (cudaMemcpy(ptr, tensor.getData().data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(ptr);
        return {};
    }

    return GpuTensor{ptr, count};
}

GpuTensor uploadLinear(const Tensor& tensor) {
    size_t out_n = tensor.getShape()[0];
    size_t in_n = tensor.getShape()[1];

    std::vector<float> host_t(in_n * out_n);
    for (size_t i = 0; i < in_n; ++i) {
        for (size_t j = 0; j < out_n; ++j) {
            host_t[i * out_n + j] = tensor[j * in_n + i]; 
        }     
    }

    float* ptr = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), in_n * out_n * sizeof(float)) != cudaSuccess) return {};
    
    if (cudaMemcpy(ptr, host_t.data(), in_n * out_n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(ptr);
        return {};
    }

    return GpuTensor{ptr, in_n * out_n};
}

bool uploadDeviceWeights(const Model& model, DeviceWeights& out) {
    freeDeviceWeights(out);

    Cursor cursor;
    cursor.params = model.parameters();

    const ModelConfig& config = model.getConfig();
    const size_t C = config.embedding_dim;
    const size_t V = config.vocab_size;
    const size_t B = config.block_size;
    const size_t d = C / config.n_heads;
    const size_t F = 4 * C;

    auto accept = [&](GpuTensor& slot, GpuTensor uploaded) {
        slot = uploaded;
        if (slot.ptr == nullptr) {
            freeDeviceWeights(out);
            return false;
        }
        return true;
    };

    try {
        if (!accept(out.token_embed, uploadPlain(cursor.next({V, C})))) return false;
        if (!accept(out.pos_embed, uploadPlain(cursor.next({B, C})))) return false;

        out.layers.resize(config.n_layers);
        for (size_t layer = 0; layer < config.n_layers; ++layer) {
            LayerWeights& s = out.layers[layer];
            if (!accept(s.ln1_gamma, uploadPlain(cursor.next({C})))) return false;
            if (!accept(s.ln1_beta, uploadPlain(cursor.next({C})))) return false;

            s.heads.resize(config.n_heads);
            for (size_t h = 0; h < config.n_heads; ++h) {
                HeadWeights& head = s.heads[h];
                if (!accept(head.Wq, uploadLinear(cursor.next({d, d})))) return false;
                if (!accept(head.bq, uploadPlain(cursor.next({d})))) return false;
                if (!accept(head.Wk, uploadLinear(cursor.next({d, d})))) return false;
                if (!accept(head.bk, uploadPlain(cursor.next({d})))) return false;
                if (!accept(head.Wv, uploadLinear(cursor.next({d, d})))) return false;
                if (!accept(head.bv, uploadPlain(cursor.next({d})))) return false;
                if (!accept(head.Wo, uploadLinear(cursor.next({d, d})))) return false;
                if (!accept(head.bo, uploadPlain(cursor.next({d})))) return false;
            }

            if (!accept(s.Wo, uploadLinear(cursor.next({C, C})))) return false;
            if (!accept(s.bo, uploadPlain(cursor.next({C})))) return false;
            if (!accept(s.ln2_gamma, uploadPlain(cursor.next({C})))) return false;
            if (!accept(s.ln2_beta, uploadPlain(cursor.next({C})))) return false;
            if (!accept(s.W1, uploadLinear(cursor.next({F, C})))) return false;
            if (!accept(s.b1, uploadPlain(cursor.next({F})))) return false;
            if (!accept(s.W2, uploadLinear(cursor.next({C, F})))) return false;
            if (!accept(s.b2, uploadPlain(cursor.next({C})))) return false;

            auto pack = [&](GpuTensor& slot, size_t one, auto take) {
                size_t count = s.heads.size() * one;
                float* ptr = nullptr;
                if (cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)) != cudaSuccess) {
                    freeDeviceWeights(out);
                    return false;
                }

                for (size_t h = 0; h < s.heads.size(); ++h) {
                    float* src = take(s.heads[h]).ptr;
                    if (cudaMemcpy(ptr + h * one, src, one * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
                        cudaFree(ptr);
                        freeDeviceWeights(out);
                        return false;
                    }
                }

                slot = GpuTensor{ptr, count};
                
                return true;
            };

            if (!pack(s.Wq_all, d * d, [](const HeadWeights& h) -> const GpuTensor& { return h.Wq; })) return false;
            if (!pack(s.bq_all, d, [](const HeadWeights& h) -> const GpuTensor& { return h.bq; })) return false;
            if (!pack(s.Wk_all, d * d, [](const HeadWeights& h) -> const GpuTensor& { return h.Wk; })) return false;
            if (!pack(s.bk_all, d, [](const HeadWeights& h) -> const GpuTensor& { return h.bk; })) return false;
            if (!pack(s.Wv_all, d * d, [](const HeadWeights& h) -> const GpuTensor& { return h.Wv; })) return false;
            if (!pack(s.bv_all, d, [](const HeadWeights& h) -> const GpuTensor& { return h.bv; })) return false;
            if (!pack(s.Wo_all, d * d, [](const HeadWeights& h) -> const GpuTensor& { return h.Wo; })) return false;
            if (!pack(s.bo_all, d, [](const HeadWeights& h) -> const GpuTensor& { return h.bo; })) return false;
        }

        if (!accept(out.ln_f_gamma, uploadPlain(cursor.next({C})))) return false;
        if (!accept(out.ln_f_beta, uploadPlain(cursor.next({C})))) return false;
        if (!accept(out.W_vocab, uploadLinear(cursor.next({V, C})))) return false;
        if (!accept(out.b_vocab, uploadPlain(cursor.next({V})))) return false;
    } catch (...) {
        freeDeviceWeights(out);
        throw;
    }

    if (cursor.index != cursor.params.size()) {
        freeDeviceWeights(out);
        throw std::runtime_error("nije se proslo kroz sve tenzore");
    }

    return true;
}

void freeGpu(GpuTensor& tensor) {
    if (tensor.ptr != nullptr) {
        cudaFree(tensor.ptr);
        tensor.ptr = nullptr;
        tensor.count = 0;
    }
}

void freeDeviceWeights(DeviceWeights& weights) {
    freeGpu(weights.token_embed);
    freeGpu(weights.pos_embed);

    for (LayerWeights& layer : weights.layers) {
        freeGpu(layer.ln1_gamma);
        freeGpu(layer.ln1_beta);

        for (HeadWeights& head : layer.heads) {
            freeGpu(head.Wq);
            freeGpu(head.bq);
            freeGpu(head.Wk);
            freeGpu(head.bk);
            freeGpu(head.Wv);
            freeGpu(head.bv);
            freeGpu(head.Wo);
            freeGpu(head.bo);
        }

        freeGpu(layer.Wo);
        freeGpu(layer.bo);
        freeGpu(layer.ln2_gamma);
        freeGpu(layer.ln2_beta);
        freeGpu(layer.W1);
        freeGpu(layer.b1);
        freeGpu(layer.W2);
        freeGpu(layer.b2);

        freeGpu(layer.Wq_all);
        freeGpu(layer.bq_all);
        freeGpu(layer.Wk_all);
        freeGpu(layer.bk_all);
        freeGpu(layer.Wv_all);
        freeGpu(layer.bv_all);
        freeGpu(layer.Wo_all);
        freeGpu(layer.bo_all);
    }

    freeGpu(weights.ln_f_gamma);
    freeGpu(weights.ln_f_beta);
    freeGpu(weights.W_vocab);
    freeGpu(weights.b_vocab);
}

bool allocDeviceWorkspace(const ModelConfig& config, DeviceWorkspace& ws) {
    freeDeviceWorkspace(ws);

    int C = config.embedding_dim;
    int n = config.n_heads;
    if (n == 0 || C % n != 0) return false;

    int d = C / n;
    int F = 4 * C;
    int capacity = config.block_size;
    int V = config.vocab_size;

    auto reserve = [](auto*& dest, size_t bytes) {
        return cudaMalloc(reinterpret_cast<void**>(&dest), bytes) == cudaSuccess;
    };

    if (!reserve(ws.ids, capacity * sizeof(int))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.x, capacity * C * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.residual, capacity * C * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.scratch, capacity * C * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.head, n * capacity * d * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.q, n * capacity * d * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.k, n * capacity * d * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.v, n * capacity * d * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.ctx, n * capacity * d * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.kt, n * capacity * d * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.scores, n * capacity * capacity * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.hidden, capacity * F * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.row, C * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (!reserve(ws.logits, V * sizeof(float))) {
        freeDeviceWorkspace(ws);
        return false;
    }

    ws.capacity = capacity;
    ws.C = C;
    ws.V = V;
    ws.d = d;
    ws.F = F;
    ws.n_heads = n;

    return true;
}

void freeDeviceWorkspace(DeviceWorkspace& ws) {
    if (ws.ids != nullptr) {
        cudaFree(ws.ids);
        ws.ids = nullptr;
    }
    if (ws.x != nullptr) {
        cudaFree(ws.x);
        ws.x = nullptr;
    }
    if (ws.residual != nullptr) {
        cudaFree(ws.residual);
        ws.residual = nullptr;
    }
    if (ws.scratch != nullptr) {
        cudaFree(ws.scratch);
        ws.scratch = nullptr;
    }
    if (ws.head != nullptr) {
        cudaFree(ws.head);
        ws.head = nullptr;
    }
    if (ws.q != nullptr) {
        cudaFree(ws.q);
        ws.q = nullptr;
    }
    if (ws.k != nullptr) {
        cudaFree(ws.k);
        ws.k = nullptr;
    }
    if (ws.v != nullptr) {
        cudaFree(ws.v);
        ws.v = nullptr;
    }
    if (ws.ctx != nullptr) {
        cudaFree(ws.ctx);
        ws.ctx = nullptr;
    }
    if (ws.kt != nullptr) {
        cudaFree(ws.kt);
        ws.kt = nullptr;
    }
    if (ws.scores != nullptr) {
        cudaFree(ws.scores);
        ws.scores = nullptr;
    }
    if (ws.hidden != nullptr) {
        cudaFree(ws.hidden);
        ws.hidden = nullptr;
    }
    if (ws.row != nullptr) {
        cudaFree(ws.row);
        ws.row = nullptr;
    }
    if (ws.logits != nullptr) {
        cudaFree(ws.logits);
        ws.logits = nullptr;
    }

    ws.capacity = 0;
    ws.C = 0;
    ws.V = 0;
    ws.d = 0;
    ws.F = 0;
    ws.n_heads = 0;
}

bool allocDeviceBackwardWs(
    size_t C,
    size_t n_heads,
    size_t capacity,
    DeviceBackwardWs& ws
) {
    freeDeviceBackwardWs(ws);

    if (n_heads == 0 || C % n_heads != 0) return false;

    const size_t d = C / n_heads;
    const size_t heads = n_heads * capacity * d;
    const size_t scores = n_heads * capacity * capacity;
    const size_t full = capacity * C;

    auto reserve = [](auto*& dest, size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&dest), count * sizeof(float)) == cudaSuccess;
    };

    if (!reserve(ws.x, heads) 
        || !reserve(ws.q, heads) 
        || !reserve(ws.k, heads) 
        || !reserve(ws.v, heads) 
        || !reserve(ws.kt, heads) 
        || !reserve(ws.ctx, heads)
    ) {
        freeDeviceBackwardWs(ws);
        return false;
    }
    if (!reserve(ws.d_ctx, heads) 
        || !reserve(ws.d_v, heads) 
        || !reserve(ws.d_q, heads) 
        || !reserve(ws.d_k, heads) 
        || !reserve(ws.d_x, heads)
    ) {
        freeDeviceBackwardWs(ws);
        return false;
    }
    if (!reserve(ws.scores, scores) 
        || !reserve(ws.d_p, scores) 
        || !reserve(ws.d_scores, scores)
    ) {
        freeDeviceBackwardWs(ws);
        return false;
    }
    if (!reserve(ws.joined, full) 
        || !reserve(ws.grad, full) 
        || !reserve(ws.d_joined, full) 
        || !reserve(ws.d_input, full)
    ) {
        freeDeviceBackwardWs(ws);
        return false;
    }

    ws.capacity = capacity;
    ws.C = C;
    ws.d = d;
    ws.n_heads = n_heads;
    return true;
}

void freeDeviceBackwardWs(DeviceBackwardWs& ws) {
    if (ws.x != nullptr) { cudaFree(ws.x); ws.x = nullptr; }
    if (ws.q != nullptr) { cudaFree(ws.q); ws.q = nullptr; }
    if (ws.k != nullptr) { cudaFree(ws.k); ws.k = nullptr; }
    if (ws.v != nullptr) { cudaFree(ws.v); ws.v = nullptr; }
    if (ws.kt != nullptr) { cudaFree(ws.kt); ws.kt = nullptr; }
    if (ws.scores != nullptr) { cudaFree(ws.scores); ws.scores = nullptr; }
    if (ws.ctx != nullptr) { cudaFree(ws.ctx); ws.ctx = nullptr; }
    if (ws.joined != nullptr) { cudaFree(ws.joined); ws.joined = nullptr; }
    if (ws.grad != nullptr) { cudaFree(ws.grad); ws.grad = nullptr; }
    if (ws.d_joined != nullptr) { cudaFree(ws.d_joined); ws.d_joined = nullptr; }
    if (ws.d_ctx != nullptr) { cudaFree(ws.d_ctx); ws.d_ctx = nullptr; }
    if (ws.d_v != nullptr) { cudaFree(ws.d_v); ws.d_v = nullptr; }
    if (ws.d_p != nullptr) { cudaFree(ws.d_p); ws.d_p = nullptr; }
    if (ws.d_scores != nullptr) { cudaFree(ws.d_scores); ws.d_scores = nullptr; }
    if (ws.d_q != nullptr) { cudaFree(ws.d_q); ws.d_q = nullptr; }
    if (ws.d_k != nullptr) { cudaFree(ws.d_k); ws.d_k = nullptr; }
    if (ws.d_x != nullptr) { cudaFree(ws.d_x); ws.d_x = nullptr; }
    if (ws.d_input != nullptr) { cudaFree(ws.d_input); ws.d_input = nullptr; }

    ws.capacity = 0;
    ws.C = 0;
    ws.d = 0;
    ws.n_heads = 0;
}

void freeAttentionState(AttentionState& state) {
    auto release = [](float*& ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    auto drop = [&](float*& ptr) {
        if (state.owns_weights) {
            release(ptr);
        } else {
            ptr = nullptr;
        }
    };
    release(state.x);
    release(state.q);
    release(state.k);
    release(state.v);
    release(state.scores);
    release(state.ctx);
    release(state.joined);
    drop(state.Wq);
    drop(state.bq);
    drop(state.Wk);
    drop(state.bk);
    drop(state.Wv);
    drop(state.bv);
    drop(state.Wo_h);
    drop(state.bo_h);
    drop(state.Wo);
    drop(state.bo);
    state.owns_weights = false;
    state.T = 0;
    state.C = 0;
    state.d = 0;
    state.n_heads = 0;
}

namespace {

void releaseDevice(float*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

void releaseAttentionWeights(AttentionState& state) {
    if (state.owns_weights) {
        releaseDevice(state.Wq);
        releaseDevice(state.bq);
        releaseDevice(state.Wk);
        releaseDevice(state.bk);
        releaseDevice(state.Wv);
        releaseDevice(state.bv);
        releaseDevice(state.Wo_h);
        releaseDevice(state.bo_h);
        releaseDevice(state.Wo);
        releaseDevice(state.bo);
    } else {
        state.Wq = nullptr;
        state.bq = nullptr;
        state.Wk = nullptr;
        state.bk = nullptr;
        state.Wv = nullptr;
        state.bv = nullptr;
        state.Wo_h = nullptr;
        state.bo_h = nullptr;
        state.Wo = nullptr;
        state.bo = nullptr;
    }
    state.owns_weights = false;
}

void releaseAttentionActivations(AttentionState& state) {
    releaseDevice(state.x);
    releaseDevice(state.q);
    releaseDevice(state.k);
    releaseDevice(state.v);
    releaseDevice(state.scores);
    releaseDevice(state.ctx);
    releaseDevice(state.joined);
    state.T = 0;
}

bool reserveDevice(float*& dest, size_t count) {
    dest = nullptr;
    return cudaMalloc(reinterpret_cast<void**>(&dest), count * sizeof(float)) == cudaSuccess;
}

bool copyLinear(const Tensor& tensor, float* dest) {
    const size_t out_n = tensor.getShape()[0];
    const size_t in_n = tensor.getShape()[1];
    std::vector<float> host(in_n * out_n);
    for (size_t i = 0; i < in_n; ++i) {
        for (size_t j = 0; j < out_n; ++j) {
            host[i * out_n + j] = tensor[j * in_n + i];
        }
    }
    return cudaMemcpy(dest, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool copyPlain(const Tensor& tensor, float* dest) {
    return cudaMemcpy(dest, tensor.getData().data(), tensor.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

}

bool uploadAttentionWeights(MultiHeadAttention& layer, AttentionState& state) {
    const std::vector<Tensor*> params = layer.parameters();
    if (params.size() < 2 || (params.size() - 2) % 8 != 0) {
        return false;
    }

    const size_t H = (params.size() - 2) / 8;
    const Tensor& big_w = *params[params.size() - 2];
    const Tensor& big_b = *params[params.size() - 1];
    if (H == 0 || big_w.getShape().size() != 2 || big_w.getShape()[0] != big_w.getShape()[1]) {
        return false;
    }

    const size_t C = big_w.getShape()[0];
    if (C % H != 0 || C > 1024 || big_b.size() != C) {
        return false;
    }
    const size_t d = C / H;
    if (d == 0 || d > 1024) {
        return false;
    }

    const bool reuse = state.owns_weights
        && state.Wq != nullptr && state.Wk != nullptr && state.Wv != nullptr && state.Wo_h != nullptr
        && state.bq != nullptr && state.bk != nullptr && state.bv != nullptr && state.bo_h != nullptr
        && state.Wo != nullptr && state.bo != nullptr
        && state.n_heads == H && state.d == d && state.C == C;

    if (!reuse) {
        releaseAttentionWeights(state);
        state.owns_weights = true;
        const size_t weight_n = H * d * d;
        const size_t bias_n = H * d;
        if (!reserveDevice(state.Wq, weight_n) || !reserveDevice(state.Wk, weight_n)
            || !reserveDevice(state.Wv, weight_n) || !reserveDevice(state.Wo_h, weight_n)
            || !reserveDevice(state.bq, bias_n) || !reserveDevice(state.bk, bias_n)
            || !reserveDevice(state.bv, bias_n) || !reserveDevice(state.bo_h, bias_n)
            || !reserveDevice(state.Wo, C * C) || !reserveDevice(state.bo, C)) {
            releaseAttentionWeights(state);
            return false;
        }
        state.owns_weights = true;
        state.n_heads = H;
        state.d = d;
        state.C = C;
    }

    for (size_t h = 0; h < H; ++h) {
        const size_t base = h * 8;
        const Tensor& wq = *params[base];
        const Tensor& bq = *params[base + 1];
        const Tensor& wk = *params[base + 2];
        const Tensor& bk = *params[base + 3];
        const Tensor& wv = *params[base + 4];
        const Tensor& bv = *params[base + 5];
        const Tensor& wo = *params[base + 6];
        const Tensor& bo = *params[base + 7];
        if (wq.getShape() != std::vector<size_t>{d, d}
            || wk.getShape() != wq.getShape()
            || wv.getShape() != wq.getShape()
            || wo.getShape() != wq.getShape()
            || bq.size() != d || bk.size() != d || bv.size() != d || bo.size() != d) {
            return false;
        }
        if (!copyLinear(wq, state.Wq + h * d * d) || !copyPlain(bq, state.bq + h * d)
            || !copyLinear(wk, state.Wk + h * d * d) || !copyPlain(bk, state.bk + h * d)
            || !copyLinear(wv, state.Wv + h * d * d) || !copyPlain(bv, state.bv + h * d)
            || !copyLinear(wo, state.Wo_h + h * d * d) || !copyPlain(bo, state.bo_h + h * d)) {
            return false;
        }
    }
    return copyLinear(big_w, state.Wo) && copyPlain(big_b, state.bo);
}

bool prepareAttentionActivations(AttentionState& state, size_t H, size_t T, size_t C, size_t d) {
    if (H == 0 || T == 0 || C == 0 || d == 0 || T > 1024 || C > 1024 || d > 1024 || C != H * d) {
        return false;
    }

    if (state.x != nullptr && state.q != nullptr && state.k != nullptr && state.v != nullptr
        && state.scores != nullptr && state.ctx != nullptr && state.joined != nullptr
        && state.T == T && state.C == C && state.d == d && state.n_heads == H) {
        return true;
    }

    releaseAttentionActivations(state);
    const size_t heads = H * T * d;
    if (!reserveDevice(state.x, heads) || !reserveDevice(state.q, heads) || !reserveDevice(state.k, heads)
        || !reserveDevice(state.v, heads) || !reserveDevice(state.ctx, heads)
        || !reserveDevice(state.scores, H * T * T) || !reserveDevice(state.joined, T * C)) {
        releaseAttentionActivations(state);
        return false;
    }

    state.T = T;
    state.C = C;
    state.d = d;
    state.n_heads = H;
    return true;
}

bool prepareAttentionState(AttentionState& state, size_t H, size_t T, size_t C, size_t d) {
    return prepareAttentionActivations(state, H, T, C, d);
}

bool copyDevice(float* dest, const float* src, size_t count) {
    return cudaMemcpy(dest, src, count * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
}

