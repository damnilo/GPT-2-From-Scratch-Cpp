
#include "../include/Generate.h"
#include "../include/Kernels.cuh"
#include "../include/Linear.h"

#include <cmath>

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

        cudaGather3d<<<dim3(T, H), ws.d>>>(ws.scratch, ws.head, H, T, ws.C, ws.d);
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
        
        if (!linearBatchForwardDevice(path, ws.head, l.Wq_all.ptr, l.bq_all.ptr, ws.q, H, T, ws.d, ws.d)) return false;
        if (!linearBatchForwardDevice(path, ws.head, l.Wk_all.ptr, l.bk_all.ptr, ws.k, H, T, ws.d, ws.d)) return false;
        if (!linearBatchForwardDevice(path, ws.head, l.Wv_all.ptr, l.bv_all.ptr, ws.v, H, T, ws.d, ws.d)) return false;
        if (slot != nullptr) {
            const size_t heads = H * T * ws.d;
            if (!copyDevice(slot->q, ws.q, heads) || !copyDevice(slot->k, ws.k, heads) || !copyDevice(slot->v, ws.v, heads)) return false;
        }

        cudaTranspose3d<<<dim3(T, H), ws.d>>>(ws.k, ws.kt, H, T, ws.d);
        if (!gemmBatchDevice(path, ws.q, ws.kt, ws.scores, H, T, ws.d, T)) return false;
        cudaMaskSoftmax3d<<<dim3(T, H), 1>>>(ws.scores, H, T, 1.0f / sqrtf(static_cast<float>(ws.d)));
        if (slot != nullptr && !copyDevice(slot->scores, ws.scores, H * T * T)) return false;

        if (!gemmBatchDevice(path, ws.scores, ws.v, ws.ctx, H, T, T, ws.d)) return false;
        if (slot != nullptr && !copyDevice(slot->ctx, ws.ctx, H * T * ws.d)) return false;
        if (!linearBatchForwardDevice(path, ws.ctx, l.Wo_all.ptr, l.bo_all.ptr, ws.head, H, T, ws.d, ws.d)) return false;
        cudaScatter3d<<<dim3(T, H), ws.d>>>(ws.head, ws.x, H, T, ws.C, ws.d);
        if (slot != nullptr && !copyDevice(slot->joined, ws.x, T * ws.C)) return false;

        if (!linearForwardDevice(path, ws.x, l.Wo.ptr, l.bo.ptr, ws.scratch, T, ws.C, ws.C)) return false;
        cudaAdd<<<T, ws.C>>>(ws.residual, ws.scratch, ws.x, T, ws.C);
        
        if (cudaMemcpy(ws.residual, ws.x, T * ws.C * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }

        cudaLayernorm<<<T, ws.C>>>(ws.x, ws.scratch, l.ln2_gamma, l.ln2_beta, T, ws.C);
        if (!linearForwardDevice(path, ws.scratch, l.W1.ptr, l.b1.ptr, ws.hidden, T, ws.C, ws.F)) return false;
        cudaGELU<<<T, ws.F>>>(ws.hidden, T, ws.F);

        if (!linearForwardDevice(path, ws.hidden, l.W2.ptr, l.b2.ptr, ws.scratch, T, ws.F, ws.C)) return false;
        cudaAdd<<<T, ws.C>>>(ws.residual, ws.scratch, ws.x, T, ws.C);
    }

    cudaLayernorm<<<1, ws.C>>>(ws.x + (T - 1) * ws.C, ws.row, weights.ln_f_gamma, weights.ln_f_beta, 1, ws.C);
    if (!linearForwardDevice(path, ws.row, weights.W_vocab.ptr, weights.b_vocab.ptr, ws.logits, 1, ws.C, ws.V)) return false;

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
