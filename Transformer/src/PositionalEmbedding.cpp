//
// Created by HP on 10/10/2026.
//
#include "../include/PositionalEmbedding.h"

PositionalEmbedding::PositionalEmbedding(size_t vocab_size, size_t block_size, size_t embedding_dim) :
tokens(vocab_size, embedding_dim),
positions(block_size, embedding_dim)
{}

Tensor PositionalEmbedding::forward(const Tensor &input) {
    Tensor token_tensor = tokens.forward(input);

    const auto& shape = input.getShape();
    const size_t sequence_len = shape.back();
    const size_t rows = shape.size() == 2 ? shape[0] : 1;

    std::vector<float> pos_input;
    pos_input.reserve(rows * sequence_len);

    for (size_t row = 0; row < rows; row++) {
        for (size_t i = 0; i < sequence_len; i++) {
            pos_input.push_back(static_cast<float>(i));
        }
    }

    Tensor tensor(shape, pos_input);
    Tensor position_tensor = positions.forward(tensor);

    return token_tensor + position_tensor;
}

Tensor PositionalEmbedding::backward(const Tensor &grad_output) {
    Tensor token_tensor = tokens.backward(grad_output);
    Tensor position_tensor = positions.backward(grad_output);

    return token_tensor + position_tensor;
}

std::vector<Tensor *> PositionalEmbedding::parameters() {
    std::vector<Tensor*> ret;

    for (Tensor* t : tokens.parameters()) {
        ret.push_back(t);
    }

    for (Tensor* t : positions.parameters()) {
        ret.push_back(t);
    }

    return ret;
}

std::vector<Tensor *> PositionalEmbedding::gradients() {
    std::vector<Tensor*> ret;

    for (Tensor* t : tokens.gradients()) {
        ret.push_back(t);
    }

    for (Tensor* t : positions.gradients()) {
        ret.push_back(t);
    }

    return ret;
}

void PositionalEmbedding::zeroGrad() {
    tokens.zeroGrad();
    positions.zeroGrad();
}
