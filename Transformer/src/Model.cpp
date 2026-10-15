//
// Created by HP on 10/10/2026.
//

#include "../include/Model.h"

#include "../../NeuralNet/Layers/include/LayerNorm.h"
#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Layers/include/Softmax.h"
#include "../include/PositionalEmbedding.h"
#include "../include/Transformer.h"

#include <stdexcept>

Model::Model(size_t vocab_size, size_t block_size, size_t embedding_dim, size_t n_heads, size_t n_layers, size_t batch_size, size_t sequence_len)
    : config{vocab_size, block_size, embedding_dim, n_heads, n_layers, batch_size, sequence_len} {
    if (n_layers == 0) {
        throw std::invalid_argument("n_layers must be at least 1");
    }
    if (n_heads == 0 || embedding_dim % n_heads != 0) {
        throw std::invalid_argument("n_heads must divide embedding_dim");
    }

    model.addLayer(std::make_unique<PositionalEmbedding>(vocab_size, block_size, embedding_dim));
    for (size_t layer = 0; layer < n_layers; layer++) {
        model.addLayer(std::make_unique<Transformer>(vocab_size, n_heads, batch_size, sequence_len, embedding_dim));
    }
    model.addLayer(std::make_unique<LayerNorm>(embedding_dim));
    model.addLayer(std::make_unique<Linear>(embedding_dim, vocab_size));
    model.addLayer(std::make_unique<Softmax>(-1));
}

const ModelConfig& Model::getConfig() const {
    return config;
}

Tensor Model::forward(const Tensor &input) const {
    return model.forward(input);
}

Tensor Model::backward(const Tensor &grad_output) {
    return model.backward(grad_output);
}

std::vector<Tensor *> Model::parameters() const {
    return model.parameters();
}

std::vector<Tensor *> Model::gradients() const {
    return model.gradients();
}

void Model::zeroGrad() const {
    model.zeroGrad();
}
