
#include "../include/DeviceForward.h"
#include "../include/DeviceBackward.h"

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

__global__ void cudaEmbed(
    int* ids, 
    GpuTensor token_embed, 
    GpuTensor pos_embed, 
    float* x, 
    size_t T, 
    size_t C
) {
    int t = blockIdx.x;
    int c = threadIdx.x;

    if (t >= T || c >= C) return;

    int id = ids[t];
    x[t * C + c] = token_embed.ptr[id * C + c] + pos_embed.ptr[t * C + c];
}

__global__ void cudaLayernorm(
    float* x, 
    float* y, 
    GpuTensor gamma, 
    GpuTensor beta, 
    size_t T, 
    size_t C
) {
    int t = blockIdx.x;
    int c = threadIdx.x;

    if (t >= T || c >= C) return;

    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += x[t * C + i];
    }

    float mean = (float)sum / C;

    sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += powf(x[t * C + i] - mean, 2);
    }

    float var = (float)sum / C;
    float inv = rsqrtf(var + 1e-5f);
    
    y[t * C + c] = gamma.ptr[c] * (x[t * C + c] - mean) * inv + beta.ptr[c];
}

__global__ void cudaGather(
    float* scratch,
    float* head,
    size_t h,
    size_t T,
    size_t C,
    size_t d
) {
    int t = blockIdx.x;
    int k = threadIdx.x;

    if (t >= T || k >= d) return;

    head[t * d + k] = scratch[t * C + h * d + k];
}

__global__ void cudaGatherAll(
    float* scratch,
    float* heads,
    size_t H,
    size_t T,
    size_t C,
    size_t d
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int k = threadIdx.x;
    
    if (t >= T || h >= H || k >= d) return;

    heads[h * T * d + t * d + k] = scratch[t * C + h * d + k];
} 

__global__ void cudaAddBias(
    float* a,
    GpuTensor bias,
    size_t T,
    size_t d
) {
    const size_t t = blockIdx.y;
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (t >= T || k >= d) return;

    a[t * d + k] += bias.ptr[k];
}

__global__ void cudaAddBiasHeads(
    float* a,
    float* bias,
    size_t H,
    size_t T,
    size_t d
) {

    int h = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= H * T || k >= d) return;

    a[h * d + k] += bias[(h / T) * d + k];
}

__global__ void cudaTranspose(
    float* a,
    float* a_t,
    size_t T,
    size_t d
) {
    int t = blockIdx.x;
    int k = threadIdx.x;

    if (t >= T || k >= d) return;

    a_t[k * T + t] = a[t * d + k];
}

__global__ void cudaTransposeAll(
    float* a,
    float* a_t,
    size_t H,
    size_t T,
    size_t d
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int k = threadIdx.x;

    if (t >= T || h >= H || k >= d) return;

    a_t[h * T * d + k * T + t] = a[h * T * d + t * d + k];
}

__global__ void cudaMaskSoftmax(
    float* scores,
    size_t T,
    float scale
) {
    int t = blockIdx.x;

    if (t >= T) return;

    float* row = scores + t * T;

    for (size_t i = 0; i < T; ++i) {
        row[i] = (i <= t) ? row[i] * scale : -INFINITY;
    }

    float max = -INFINITY;
    for (size_t i = 0; i < T; ++i) {
        if (max < row[i]) {
            max = row[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < T; ++i) {
        sum += (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max);
    }

    for (size_t i = 0; i < T; ++i) {
        row[i] = (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max) / sum;
    }
}

__global__ void cudaMaskSoftmaxAll(
    float* scores,
    size_t H,
    size_t T,
    float scale
) {

    int t = blockIdx.x;
    int h = blockIdx.y;

    if (t >= T || h >= H) return;

    float* row = scores + (h * T + t) * T;

    for (int i = 0; i < T; ++i) {
        row[i] = (i <= t) ? row[i] * scale : -INFINITY;
    } 

    float max = -INFINITY;
    for (int i = 0; i < T; ++i) {
        if (max < row[i]) max = row[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        sum += (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max);
    }

    for (int i = 0; i < T; ++i) {
        row[i] = (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max) / sum;
    }
}

__global__ void cudaScatter(
    float* head,
    float* x,
    size_t h,
    size_t T,
    size_t C,
    size_t d
) {
    int t = blockIdx.x;
    int k = threadIdx.x;

    if (t >= T || k >= d) return;

    x[t * C + h * d + k] = head[t * d + k];
}

__global__ void cudaScatterAll(
    float* head,
    float* x,
    size_t H,
    size_t T,
    size_t C,
    size_t d
) {

    int t = blockIdx.x;
    int h = blockIdx.y;
    int k = threadIdx.x;

    if (t >= T || h >= H || k >= d) return;

    x[t * C + h * d + k] = head[h * T * d + t * d + k];
}

__global__ void cudaAdd(
    float* residual,
    float* scratch,
    float* x,
    size_t T,
    size_t C
) {
    int t = blockIdx.x;
    int c = threadIdx.x;

    if (t >= T || c >= C) return;

    x[t * C + c] = residual[t * C + c] + scratch[t * C + c];
}

__global__ void cudaGELU(
    float* x,
    size_t T,
    size_t F
) {
    int t = blockIdx.x;
    int f = threadIdx.x;

    if (t >= T || f >= F) return;

    float val = x[t * F + f];
    x[t * F + f] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + val * val * val * 0.044715f)));
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

bool prepareAttentionState(AttentionState& state, size_t H, size_t T, size_t C, size_t d) {
    if (state.x != nullptr 
        && state.q != nullptr 
        && state.k != nullptr 
        && state.v != nullptr
        && state.scores != nullptr 
        && state.ctx != nullptr 
        && state.joined != nullptr
        && state.T == T 
        && state.C == C 
        && state.d == d 
        && state.n_heads == H
        && !state.owns_weights
    ) {
        return true;
    }

    freeAttentionState(state);
    auto reserve = [](float*& dest, size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&dest), count * sizeof(float)) == cudaSuccess;
    };

    const size_t heads = H * T * d;
    
    if (!reserve(state.x, heads) 
        || !reserve(state.q, heads) 
        || !reserve(state.k, heads)
        || !reserve(state.v, heads) 
        || !reserve(state.ctx, heads)
        || !reserve(state.scores, H * T * T) 
        || !reserve(state.joined, T * C)
    ) {
        freeAttentionState(state);
        return false;
    }
    state.T = T;
    state.C = C;
    state.d = d;
    state.n_heads = H;
    return true;
}

bool copyDevice(float* dest, const float* src, size_t count) {
    return cudaMemcpy(dest, src, count * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
}

}

bool forwardAttentionDevice(
    GemmDevicePath path,
    MultiHeadAttention& layer,
    const float* host_in,
    size_t T,
    AttentionState& state
) {
    freeAttentionState(state);
    if (host_in == nullptr || T == 0 || T > 1024) return false;

    const std::vector<Tensor*> params = layer.parameters();
    if (params.size() < 2 || (params.size() - 2) % 8 != 0) return false;

    const size_t H = (params.size() - 2) / 8;
    const Tensor& big_w = *params[params.size() - 2];
    const Tensor& big_b = *params[params.size() - 1];
    if (H == 0 || big_w.getShape().size() != 2 || big_w.getShape()[0] != big_w.getShape()[1]) return false;

    const size_t C = big_w.getShape()[0];
    if (C % H != 0 || C > 1024 || big_b.size() != C) return false;
    const size_t d = C / H;
    if (d == 0 || d > 1024) return false;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    state.owns_weights = true;
    auto reserve = [](float*& dest, size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&dest), count * sizeof(float)) == cudaSuccess;
    };
    auto fail = [&]() {
        freeAttentionState(state);
        return false;
    };

    const size_t weight_n = H * d * d;
    const size_t bias_n = H * d;
    const size_t act_n = H * T * d;
    if (!reserve(state.Wq, weight_n) || !reserve(state.Wk, weight_n) || !reserve(state.Wv, weight_n) || !reserve(state.Wo_h, weight_n)) return fail();
    if (!reserve(state.bq, bias_n) || !reserve(state.bk, bias_n) || !reserve(state.bv, bias_n) || !reserve(state.bo_h, bias_n)) return fail();
    if (!reserve(state.Wo, C * C) || !reserve(state.bo, C)) return fail();
    if (!reserve(state.x, act_n) || !reserve(state.q, act_n) || !reserve(state.k, act_n) || !reserve(state.v, act_n) || !reserve(state.ctx, act_n)) return fail();
    if (!reserve(state.scores, H * T * T) || !reserve(state.joined, T * C)) return fail();

    auto uploadLinear = [](const Tensor& tensor, float* dest) {
        const size_t out_n = tensor.getShape()[0];
        const size_t in_n = tensor.getShape()[1];
        std::vector<float> host(in_n * out_n);
        for (size_t i = 0; i < in_n; ++i) {
            for (size_t j = 0; j < out_n; ++j) {
                host[i * out_n + j] = tensor[j * in_n + i];
            }
        }
        return cudaMemcpy(dest, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    };
    auto uploadPlain = [](const Tensor& tensor, float* dest) {
        return cudaMemcpy(dest, tensor.getData().data(), tensor.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    };

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
        if (wq.getShape() != std::vector<size_t>{d, d} || wk.getShape() != wq.getShape() || wv.getShape() != wq.getShape() || wo.getShape() != wq.getShape()) return fail();
        if (bq.size() != d || bk.size() != d || bv.size() != d || bo.size() != d) return fail();
        if (!uploadLinear(wq, state.Wq + h * d * d) || !uploadPlain(bq, state.bq + h * d)) return fail();
        if (!uploadLinear(wk, state.Wk + h * d * d) || !uploadPlain(bk, state.bk + h * d)) return fail();
        if (!uploadLinear(wv, state.Wv + h * d * d) || !uploadPlain(bv, state.bv + h * d)) return fail();
        if (!uploadLinear(wo, state.Wo_h + h * d * d) || !uploadPlain(bo, state.bo_h + h * d)) return fail();
    }
    if (!uploadLinear(big_w, state.Wo) || !uploadPlain(big_b, state.bo)) return fail();

    float* kt = nullptr;
    float* head_out = nullptr;
    if (!reserve(kt, act_n) || !reserve(head_out, act_n)) {
        cudaFree(kt);
        cudaFree(head_out);
        return fail();
    }
    auto fail_local = [&]() {
        cudaFree(kt);
        cudaFree(head_out);
        return fail();
    };

    if (cudaMemcpy(state.joined, host_in, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return fail_local();

    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    cudaGatherAll<<<grid, static_cast<unsigned>(d)>>>(state.joined, state.x, H, T, C, d);
    if (!gemmBatchDevice(path, state.x, state.Wq, state.q, H, T, d, d)) return fail_local();
    cudaAddBiasHeads<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(state.q, state.bq, H, T, d);
    if (!gemmBatchDevice(path, state.x, state.Wk, state.k, H, T, d, d)) return fail_local();
    cudaAddBiasHeads<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(state.k, state.bk, H, T, d);
    if (!gemmBatchDevice(path, state.x, state.Wv, state.v, H, T, d, d)) return fail_local();
    cudaAddBiasHeads<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(state.v, state.bv, H, T, d);

    cudaTransposeAll<<<grid, static_cast<unsigned>(d)>>>(state.k, kt, H, T, d);
    if (!gemmBatchDevice(path, state.q, kt, state.scores, H, T, d, T)) return fail_local();
    cudaMaskSoftmaxAll<<<grid, 1>>>(state.scores, H, T, scale);
    if (!gemmBatchDevice(path, state.scores, state.v, state.ctx, H, T, T, d)) return fail_local();
    if (!gemmBatchDevice(path, state.ctx, state.Wo_h, head_out, H, T, d, d)) return fail_local();
    cudaAddBiasHeads<<<dim3(static_cast<unsigned>((d + 255) / 256), static_cast<unsigned>(H * T)), 256>>>(head_out, state.bo_h, H, T, d);
    cudaScatterAll<<<grid, static_cast<unsigned>(d)>>>(head_out, state.joined, H, T, C, d);

    cudaFree(kt);
    cudaFree(head_out);
    state.T = T;
    state.C = C;
    state.d = d;
    state.n_heads = H;
    return true;
}

bool forwardDevice(
    GemmDevicePath path, 
    const DeviceWeights& weights, 
    DeviceWorkspace& ws, 
    const int* host_ids, 
    size_t T, 
    float* host_probs,
    AttentionState* saved
) {
    if (T == 0 
        || T > ws.capacity 
        || ws.V == 0 
        || host_ids == nullptr 
        || host_probs == nullptr
    ) {
        return false;
    }

    if (cudaMemcpy(ws.ids, host_ids, T * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    auto addBias = [](float* data, GpuTensor bias, size_t rows, size_t cols) {
        constexpr unsigned threads = 256;
        const dim3 block(threads);
        const dim3 grid((cols + threads - 1) / threads, rows);
        cudaAddBias<<<grid, block>>>(data, bias, rows, cols);
    };

    auto addBiasHeads = [](float* data, GpuTensor bias, size_t heads, size_t rows, size_t cols) {
        constexpr unsigned threads = 256;
        const dim3 block(threads);
        const dim3 grid((cols + threads - 1) / threads, heads * rows);
        cudaAddBiasHeads<<<grid, block>>>(data, bias.ptr, heads, rows, cols);
    };

    cudaEmbed<<<T, ws.C>>>(ws.ids, weights.token_embed, weights.pos_embed, ws.x, T, ws.C);

    for (size_t layer_i = 0; layer_i < weights.layers.size(); ++layer_i) {
        const LayerWeights& l = weights.layers[layer_i];
        AttentionState* slot = (saved == nullptr) ? nullptr : saved + layer_i;

        if (cudaMemcpy(ws.residual, ws.x, T * ws.C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }

        cudaLayernorm<<<T, ws.C>>>(ws.x, ws.scratch, l.ln1_gamma, l.ln1_beta, T, ws.C);

        
        const size_t H = ws.n_heads;
        if (l.heads.size() != H) return false;

        cudaGatherAll<<<dim3(T, H), ws.d>>>(ws.scratch, ws.head, H, T, ws.C, ws.d);
        if (slot != nullptr) {
            if (!prepareAttentionState(*slot, H, T, ws.C, ws.d)) return false;
            slot->Wq = l.Wq_all.ptr;
            slot->bq = l.bq_all.ptr;
            slot->Wk = l.Wk_all.ptr;
            slot->bk = l.bk_all.ptr;
            slot->Wv = l.Wv_all.ptr;
            slot->bv = l.bv_all.ptr;
            slot->Wo_h = l.Wo_all.ptr;
            slot->bo_h = l.bo_all.ptr;
            slot->Wo = l.Wo.ptr;
            slot->bo = l.bo.ptr;
            if (!copyDevice(slot->x, ws.head, H * T * ws.d)) return false;
        }
        
        if (!gemmBatchDevice(path, ws.head, l.Wq_all.ptr, ws.q, H, T, ws.d, ws.d)) return false;
        addBiasHeads(ws.q, l.bq_all, H, T, ws.d);
        if (!gemmBatchDevice(path, ws.head, l.Wk_all.ptr, ws.k, H, T, ws.d, ws.d)) return false;
        addBiasHeads(ws.k, l.bk_all, H, T, ws.d);
        if (!gemmBatchDevice(path, ws.head, l.Wv_all.ptr, ws.v, H, T, ws.d, ws.d)) return false;
        addBiasHeads(ws.v, l.bv_all, H, T, ws.d);
        if (slot != nullptr) {
            const size_t heads = H * T * ws.d;
            if (!copyDevice(slot->q, ws.q, heads) || !copyDevice(slot->k, ws.k, heads) || !copyDevice(slot->v, ws.v, heads)) return false;
        }

        cudaTransposeAll<<<dim3(T, H), ws.d>>>(ws.k, ws.kt, H, T, ws.d);
        if (!gemmBatchDevice(path, ws.q, ws.kt, ws.scores, H, T, ws.d, T)) return false;
        cudaMaskSoftmaxAll<<<dim3(T, H), 1>>>(ws.scores, H, T, 1.0f / sqrtf(static_cast<float>(ws.d)));
        if (slot != nullptr && !copyDevice(slot->scores, ws.scores, H * T * T)) return false;

        if (!gemmBatchDevice(path, ws.scores, ws.v, ws.ctx, H, T, T, ws.d)) return false;
        if (slot != nullptr && !copyDevice(slot->ctx, ws.ctx, H * T * ws.d)) return false;
        if (!gemmBatchDevice(path, ws.ctx, l.Wo_all.ptr, ws.head, H, T, ws.d, ws.d)) return false;
        addBiasHeads(ws.head, l.bo_all, H, T, ws.d);
        cudaScatterAll<<<dim3(T, H), ws.d>>>(ws.head, ws.x, H, T, ws.C, ws.d);
        if (slot != nullptr && !copyDevice(slot->joined, ws.x, T * ws.C)) return false;

        if (!gemmDevice(path, ws.x, l.Wo.ptr, ws.scratch, T, ws.C, ws.C)) return false;
        addBias(ws.scratch, l.bo, T, ws.C);
        cudaAdd<<<T, ws.C>>>(ws.residual, ws.scratch, ws.x, T, ws.C);
        
        if (cudaMemcpy(ws.residual, ws.x, T * ws.C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }

        cudaLayernorm<<<T, ws.C>>>(ws.x, ws.scratch, l.ln2_gamma, l.ln2_beta, T, ws.C);
        if (!gemmDevice(path, ws.scratch, l.W1.ptr, ws.hidden, T, ws.C, ws.F)) return false;
        addBias(ws.hidden, l.b1, T, ws.F);
        cudaGELU<<<T, ws.F>>>(ws.hidden, T, ws.F);

        if (!gemmDevice(path, ws.hidden, l.W2.ptr, ws.scratch, T, ws.F, ws.C)) return false;
        addBias(ws.scratch, l.b2, T, ws.C);
        cudaAdd<<<T, ws.C>>>(ws.residual, ws.scratch, ws.x, T, ws.C);
    }

    cudaLayernorm<<<1, ws.C>>>(ws.x + (T - 1) * ws.C, ws.row, weights.ln_f_gamma, weights.ln_f_beta, 1, ws.C);
    if (!gemmDevice(path, ws.row, weights.W_vocab.ptr, ws.logits, 1, ws.C, ws.V)) return false;
    addBias(ws.logits, weights.b_vocab, 1, ws.V);

    if (cudaMemcpy(host_probs, ws.logits, ws.V * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    float max = host_probs[0];
    for (size_t i = 1; i < ws.V; ++i) {
        if (max < host_probs[i]) max = host_probs[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < ws.V; ++i) {
        host_probs[i] = expf(host_probs[i] - max);
        sum += host_probs[i];
    }

    for (size_t i = 0; i < ws.V; ++i) {
        host_probs[i] /= sum;
    }

    return true;
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