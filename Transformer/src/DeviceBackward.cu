
#include "../include/DeviceBackward.h"

#include <cmath>
#include <cuda_runtime.h>

__global__ void cudaGatherGrad(
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

__global__ void cudaAddVectors(
    float* a,
    const float* b,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void cudaSumRows(
    float* a,
    float* bias,
    size_t H,
    size_t T, 
    size_t d
) {
    int h = blockIdx.x;
    int k = threadIdx.x;

    if (h >= H || k >= d) return;

    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        sum += a[h * d * T + i * d + k];
    }

    bias[h * d + k] = sum;
}

__global__ void cudaTransposeGrad(
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

__global__ void cudaSoftmaxBackward(
    float* scores,
    float* d_p,
    float* d_scores,
    size_t H,
    size_t T,
    float scale
) {
    int t = blockIdx.x;
    int h = blockIdx.y;

    if (t >= T || h >= H) return;

    int row = (h * T + t) * T;
    float* tmp = d_scores + row;

    for (int i = 0; i < T; ++i) {
        tmp[i] = d_p[row + i] * scores[row + i];
    }

    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        sum += tmp[i];
    }

    for (int i = 0; i < T; ++i) {
        d_scores[row + i] = scores[row + i] * (d_p[row + i] - sum) * scale;
    }
}

__global__ void cudaScatterGrad(
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

    cudaTransposeGrad<<<C, C>>>(state.Wo, w_t, 1, C, C);
    if (!gemmDevice(path, ws.grad, w_t, ws.d_joined, T, C, C)) return fail();

    cudaTransposeGrad<<<T, C>>>(state.joined, ws.x, 1, T, C);
    if (!gemmDevice(path, ws.x, ws.grad, dw, C, T, C)) return fail();

    cudaTransposeGrad<<<C, C>>>(dw, w_t, 1, C, C);
    out.Wo.resize(C * C);
    out.bo.resize(C);
    if (cudaMemcpy(out.Wo.data(), w_t, C * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }
    cudaSumRows<<<1, C>>>(ws.grad, db, 1, T, C);
    if (cudaMemcpy(out.bo.data(), db, C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaGatherGrad<<<dim3(T, H), d>>>(ws.d_joined, ws.ctx, H, T, C, d);

    cudaTransposeGrad<<<dim3(d, H), d>>>(state.Wo_h, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.ctx, w_t, ws.d_ctx, H, T, d, d)) return fail();

    cudaTransposeGrad<<<dim3(T, H), d>>>(state.ctx, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.ctx, dw, H, d, T, d)) return fail();

    out.heads.resize(H);
    for (size_t h = 0; h < H; ++h) {
        cudaTransposeGrad<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
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

    cudaTransposeGrad<<<dim3(T, H), T>>>(state.scores, ws.scores, H, T, T);
    if (!gemmBatchDevice(path, ws.scores, ws.d_ctx, ws.d_v, H, T, T, d)) return fail();

    cudaTransposeGrad<<<dim3(T, H), d>>>(state.v, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.d_ctx, ws.kt, ws.d_p, H, T, d, T)) return fail();

    cudaSoftmaxBackward<<<dim3(T, H), 1>>>(state.scores, ws.d_p, ws.d_scores, H, T, scale);
    if (!gemmBatchDevice(path, ws.d_scores, state.k, ws.d_q, H, T, T, d)) return fail();

    cudaTransposeGrad<<<dim3(T, H), T>>>(ws.d_scores, ws.scores, H, T, T);
    if (!gemmBatchDevice(path, ws.scores, state.q, ws.d_k, H, T, T, d)) return fail();

    size_t n = H * T * d;

    cudaTransposeGrad<<<dim3(d, H), d>>>(state.Wq, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.d_q, w_t, ws.d_x, H, T, d, d)) return fail();

    cudaTransposeGrad<<<dim3(T, H), d>>>(state.x, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.d_q, dw, H, d, T, d)) return fail();

    for (size_t h = 0; h < H; ++h) {
        cudaTransposeGrad<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
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
    
    cudaTransposeGrad<<<dim3(d, H), d>>>(state.Wk, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.d_k, w_t, ws.q, H, T, d, d)) return fail();

    cudaTransposeGrad<<<dim3(T, H), d>>>(state.x, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.d_k, dw, H, d, T, d)) return fail();

    for (size_t h = 0; h < H; ++h) {
        cudaTransposeGrad<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
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
    
    cudaTransposeGrad<<<dim3(d, H), d>>>(state.Wv, w_t, H, d, d);
    if (!gemmBatchDevice(path, ws.d_v, w_t, ws.q, H, T, d, d)) return fail();

    cudaTransposeGrad<<<dim3(T, H), d>>>(state.x, ws.kt, H, T, d);
    if (!gemmBatchDevice(path, ws.kt, ws.d_v, dw, H, d, T, d)) return fail();

    for (size_t h = 0; h < H; ++h) {
        cudaTransposeGrad<<<dim3(d, 1), d>>>(dw + h * d * d, w_t, 1, d, d);
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

    cudaScatterGrad<<<dim3(T, H), d>>>(ws.d_x, ws.d_input, H, T, C, d);
    out.input.resize(T * C);
    if (cudaMemcpy(out.input.data(), ws.d_input, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail();
    }

    cudaFree(w_t);
    cudaFree(dw);
    cudaFree(db);
    return true;
}