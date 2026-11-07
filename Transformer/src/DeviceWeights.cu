#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "../../NumCPP/include/Tensor.h"
#include "../include/DeviceWeights.h"

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
    }

    freeGpu(weights.ln_f_gamma);
    freeGpu(weights.ln_f_beta);
    freeGpu(weights.W_vocab);
    freeGpu(weights.b_vocab);
}