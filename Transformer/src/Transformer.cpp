//
// Created by HP on 10/9/2026.
//
#include "../include/Transformer.h"

#include "../../NeuralNet/Layers/include/LayerNorm.h"
#include "../../NeuralNet/Layers/include/Sequential.h"
#include "../include/FeedForward.h"
#include "../include/MultiHeadAttention.h"

Transformer::Transformer(size_t,
                         size_t n_heads,
                         size_t batch_size,
                         size_t sequence_len,
                         size_t embedding_dim) {
    layers.push_back(std::make_unique<LayerNorm>(embedding_dim));
    layers.push_back(std::make_unique<MultiHeadAttention>(n_heads, batch_size, sequence_len, embedding_dim));
    layers.push_back(std::make_unique<LayerNorm>(embedding_dim));
    layers.push_back(std::make_unique<FeedForward>(embedding_dim, embedding_dim));
}

Tensor Transformer::forward(const Tensor &input) {
    Tensor residual = input;
    Tensor out = layers[0]->forward(input);
    out = layers[1]->forward(out);
    out = residual + out;

    residual = out;
    out = layers[2]->forward(out);
    out = layers[3]->forward(out);
    return residual + out;
}

Tensor Transformer::backward(const Tensor &grad_output) {
    Tensor through_ffn = layers[3]->backward(grad_output);
    Tensor through_ln2 = layers[2]->backward(through_ffn);
    Tensor grad = through_ln2 + grad_output;

    Tensor through_mha = layers[1]->backward(grad);
    Tensor through_ln1 = layers[0]->backward(through_mha);
    return through_ln1 + grad;
}

std::vector<Tensor *> Transformer::parameters() {
    std::vector<Tensor*> parameters;

    for (const auto& l : this->layers) {
        for (Tensor* t : l->parameters()) {
            parameters.push_back(t);
        }
    }

    return parameters;
}

std::vector<Tensor *> Transformer::gradients() {
    std::vector<Tensor*> gradients;

    for (const auto& l : this->layers) {
        for (Tensor* t : l->gradients()) {
            gradients.push_back(t);
        }
    }

    return gradients;
}

void Transformer::zeroGrad() {
    for (const auto& l : this->layers) {
        l->zeroGrad();
    }
}

