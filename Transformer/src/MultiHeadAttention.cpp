//
// Created by HP on 10/9/2026.
//
#include "../include/MultiHeadAttention.h"
#include "../../Cuda/include/Attention.h"

#include <stdexcept>

namespace {

void writeGradient(Tensor& destination, const std::vector<float>& source) {
    if (destination.size() != source.size()) {
        throw std::runtime_error("gradient size mismatch");
    }
    destination = Tensor(destination.getShape(), source);
}

}

MultiHeadAttention::MultiHeadAttention(size_t n_heads, size_t batch_size, size_t sequence_len, size_t embedding_dim)
    : head_dim(n_heads == 0 ? 0 : embedding_dim / n_heads),
      out_linear(embedding_dim, embedding_dim) {
    if (n_heads == 0 || embedding_dim % n_heads != 0) {
        throw std::invalid_argument("embedding_dim must be divisible by n_heads");
    }

    for (size_t i = 0; i < n_heads; ++i) {
        heads.emplace_back(batch_size, sequence_len, head_dim);
    }
}

Tensor MultiHeadAttention::forward(const Tensor &input) {
    if (input.ndim() != 3 || input.getShape()[2] != head_dim * heads.size()) {
        throw std::invalid_argument("MultiHeadAttention input shape mismatch");
    }
    
    if (device != nullptr) {
        if (input.getShape()[0] != 1) throw std::runtime_error("device attention needs batch 1");

        const size_t T = input.getShape()[1];
        if (!forwardAttentionDevice(device->path, *this, input.getData().data(), T, device->state)) {
            throw std::runtime_error("attention forward on device failed");
        }


    }

    Tensor combined;

    for (size_t i = 0; i < heads.size(); ++i) {
        Tensor head_input = input.slice(-1, i * head_dim, (i + 1) * head_dim);
        Tensor head_output = heads[i].forward(head_input);

        if (i == 0) {
            combined = head_output;
        } else {
            combined = combined.concat(head_output, -1);
        }
    }

    Tensor out = out_linear.forward(combined);

    return out;
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output) {
    if (device == nullptr) {
        Tensor grad_combined = out_linear.backward(grad_output);
        Tensor grad_input;

        for (size_t i = 0; i < heads.size(); ++i) {
            Tensor head_grad = grad_combined.slice(-1, i * head_dim, (i + 1) * head_dim);
            Tensor head_input_grad = heads[i].backward(head_grad);

            if (i == 0) {
                grad_input = head_input_grad;
            } else {
                grad_input = grad_input.concat(head_input_grad, -1);
            }
        }

        return grad_input;
    }

    AttentionGrad ag;
    if (!backwardAttentionDevice(device->path, device->state, *this, device->ws, grad_output.getData().data(), ag)) {
        throw std::runtime_error("attention backward on device failed");
    }

    const size_t n_heads = heads.size();
    const std::vector<Tensor*> grads = gradients();
    const size_t expected = n_heads * 8 + 2;
    if (grads.size() != expected || ag.heads.size() != n_heads) {
        throw std::runtime_error("attention gradient layout mismatch");
    }

    for (size_t h = 0; h < n_heads; ++h) {
        const size_t base = h * 8;
        const HeadGrad& head = ag.heads[h];
        writeGradient(*grads[base], head.Wq);
        writeGradient(*grads[base + 1], head.bq);
        writeGradient(*grads[base + 2], head.Wk);
        writeGradient(*grads[base + 3], head.bk);
        writeGradient(*grads[base + 4], head.Wv);
        writeGradient(*grads[base + 5], head.bv);
        writeGradient(*grads[base + 6], head.Wo);
        writeGradient(*grads[base + 7], head.bo);
    }
    writeGradient(*grads[expected - 2], ag.Wo);
    writeGradient(*grads[expected - 1], ag.bo);

    return Tensor(grad_output.getShape(), ag.input);
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    std::vector<Tensor*> params;

    for (Attention& head : heads) {
        auto head_params = head.parameters();
        params.insert(params.end(), head_params.begin(), head_params.end());
    }

    auto output_params = out_linear.parameters();
    params.insert(params.end(), output_params.begin(), output_params.end());
    return params;
}

std::vector<Tensor*> MultiHeadAttention::gradients() {
    std::vector<Tensor*> grads;

    for (Attention& head : heads) {
        auto head_grads = head.gradients();
        grads.insert(grads.end(), head_grads.begin(), head_grads.end());
    }

    auto output_grads = out_linear.gradients();
    grads.insert(grads.end(), output_grads.begin(), output_grads.end());
    return grads;
}

void MultiHeadAttention::zeroGrad() {
    for (Attention& head : heads) {
        head.zeroGrad();
    }

    out_linear.zeroGrad();
}

void MultiHeadAttention::useDevice(AttentionDevice* slot) {
    device = slot;
};