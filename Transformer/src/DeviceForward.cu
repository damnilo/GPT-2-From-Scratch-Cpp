
#include "../include/DeviceForward.h"

#include <cmath>
#include <cuda_runtime.h>

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

__global__ void cudaMaskSoftmax(
    float* scores,
    size_t T,
    float scale
) {
    int t = blockIdx.x;

    if (t >= T) return;

    for (size_t i = 0; i < T; ++i) {
        scores[t * T + i] = (i <= t) ? scores[t * T + i] * scale : -INFINITY;
    }

    float max = -INFINITY;
    for (size_t i = 0; i < T; ++i) {
        if (max < scores[t * T + i]) {
            max = scores[t * T + i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < T; ++i) {
        sum += (scores[t * T + i] == -INFINITY) ? 0.0f : expf(scores[t * T + i] - max);
    }

    for (size_t i = 0; i < T; ++i) {
        scores[t * T + i] = (scores[t * T + i] == -INFINITY) ? 0.0f : expf(scores[t * T + i] - max) / sum;
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

bool forwardDevice(
    GemmDevicePath path, 
    const DeviceWeights& weights, 
    DeviceWorkspace& ws, 
    const int* host_ids, 
    size_t T, 
    float* host_probs
) {
    if (T == 0 || T > ws.capacity || ws.V == 0 || host_ids == nullptr || host_probs == nullptr) {
        return false;
    }

    if (cudaMemcpy(ws.ids, host_ids, T * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    auto addBias = [](float* data, GpuTensor bias, size_t rows, size_t cols) {
        constexpr unsigned threads = 256;
        const dim3 block(threads);
        const dim3 grid(static_cast<unsigned>((cols + threads - 1) / threads), static_cast<unsigned>(rows));
        cudaAddBias<<<grid, block>>>(data, bias, rows, cols);
    };

    cudaEmbed<<<T, ws.C>>>(ws.ids, weights.token_embed, weights.pos_embed, ws.x, T, ws.C);

    for (const auto& l : weights.layers) {
        if (cudaMemcpy(ws.residual, ws.x, T * ws.C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }

        cudaLayernorm<<<T, ws.C>>>(ws.x, ws.scratch, l.ln1_gamma, l.ln1_beta, T, ws.C);

        for (size_t h = 0; h < l.heads.size(); ++h) {
            const HeadWeights& head = l.heads[h];
            cudaGather<<<T, ws.d>>>(ws.scratch, ws.head, h, T, ws.C, ws.d);
            if (!gemmDevice(path, ws.head, head.Wq.ptr, ws.q, T, ws.d, ws.d)) return false;
            addBias(ws.q, head.bq, T, ws.d);
            if (!gemmDevice(path, ws.head, head.Wk.ptr, ws.k, T, ws.d, ws.d)) return false;
            addBias(ws.k, head.bk, T, ws.d);
            if (!gemmDevice(path, ws.head, head.Wv.ptr, ws.v, T, ws.d, ws.d)) return false;
            addBias(ws.v, head.bv, T, ws.d);

            cudaTranspose<<<T, ws.d>>>(ws.k, ws.kt, T, ws.d);
            if (!gemmDevice(path, ws.q, ws.kt, ws.scores, T, ws.d, T)) return false;
            cudaMaskSoftmax<<<T, 1>>>(ws.scores, T, 1.0f / sqrtf(static_cast<float>(ws.d)));

            if (!gemmDevice(path, ws.scores, ws.v, ws.ctx, T, T, ws.d)) return false;
            if (!gemmDevice(path, ws.ctx, head.Wo.ptr, ws.head, T, ws.d, ws.d)) return false;
            addBias(ws.head, head.bo, T, ws.d);
            cudaScatter<<<T, ws.d>>>(ws.head, ws.x, h, T, ws.C, ws.d);
        }

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
    if (config.n_heads == 0 || C % config.n_heads != 0) return false;

    int d = C / config.n_heads;
    int F = 4 * C;
    int capacity = config.block_size;
    int V = config.vocab_size;

    if (cudaMalloc(&ws.ids, capacity * sizeof(int)) != cudaSuccess){ 
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.x, capacity * C * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.residual, capacity * C * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.scratch, capacity * C * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.head, capacity * d * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.q, capacity * d * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.k, capacity * d * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.v, capacity * d * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.ctx, capacity * d * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.kt, capacity * d * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.scores, capacity * capacity * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.hidden, capacity * F * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.row, C * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }
    if (cudaMalloc(&ws.logits, V * sizeof(float)) != cudaSuccess) {
        freeDeviceWorkspace(ws);
        return false;
    }

    ws.capacity = capacity;
    ws.C = C;
    ws.V = V;
    ws.d = d;
    ws.F = F;

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
}