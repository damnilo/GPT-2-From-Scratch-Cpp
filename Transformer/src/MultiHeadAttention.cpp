//
// Created by HP on 10/9/2026.
//
#include "../include/MultiHeadAttention.h"

#include <stdexcept>

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

    return out_linear.forward(combined);
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output) {
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
