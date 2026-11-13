//
// Created by HP on 10/10/2026.
//

#include "../include/Model.h"

#include "../../Cuda/include/Block.h"
#include "../../NeuralNet/Layers/include/LayerNorm.h"
#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Layers/include/Softmax.h"
#include "../include/PositionalEmbedding.h"
#include "../include/Transformer.h"

#include <stdexcept>
#include <vector>

Model::Model(size_t vocab_size, size_t block_size, size_t embedding_dim, size_t n_heads, size_t n_layers, size_t batch_size, size_t sequence_len)
    : config{vocab_size, block_size, embedding_dim, n_heads, n_layers, batch_size, sequence_len} {
    if (n_layers == 0) {
        throw std::invalid_argument("n_layers must be at least 1");
    }
    if (n_heads == 0 || embedding_dim % n_heads != 0) {
        throw std::invalid_argument("n_heads must divide embedding_dim");
    }

    auto tokens = std::make_unique<PositionalEmbedding>(vocab_size, block_size, embedding_dim);
    embedding = tokens.get();
    model.addLayer(std::move(tokens));
    for (size_t layer = 0; layer < n_layers; layer++) {
        auto block = std::make_unique<Transformer>(vocab_size, n_heads, batch_size, sequence_len, embedding_dim);
        blocks.push_back(block.get());
        model.addLayer(std::move(block));
    }
    auto norm = std::make_unique<LayerNorm>(embedding_dim);
    final_ln = norm.get();
    model.addLayer(std::move(norm));
    auto projection = std::make_unique<Linear>(embedding_dim, vocab_size);
    vocab = projection.get();
    model.addLayer(std::move(projection));
    auto probabilities = std::make_unique<Softmax>(-1);
    softmax = probabilities.get();
    model.addLayer(std::move(probabilities));
}

const ModelConfig& Model::getConfig() const {
    return config;
}

Tensor Model::forward(const Tensor &input) const {
    if (stack == nullptr) {
        return model.forward(input);
    }
    if (input.ndim() != 2 || input.getShape()[0] != 1) {
        throw std::runtime_error("device stack needs batch 1");
    }
    const size_t T = input.getShape()[1];
    std::vector<int> ids(T);
    for (size_t t = 0; t < T; ++t) {
        const float raw = input[t];
        if (raw < 0.0f) {
            throw std::runtime_error("device stack forward failed");
        }
        const auto id = static_cast<size_t>(raw);
        if (id >= config.vocab_size) {
            throw std::runtime_error("device stack forward failed");
        }
        ids[t] = static_cast<int>(id);
    }
    if (!forwardStackDevice(*stack, *this, ids.data(), T)) {
        throw std::runtime_error("device stack forward failed");
    }
    return Tensor(std::vector<size_t>{0});
}

Tensor Model::backward(const Tensor &grad_output) {
    if (stack == nullptr) {
        return model.backward(grad_output);
    }
    if (!backwardStackDevice(*stack, *this, nullptr)) {
        throw std::runtime_error("device stack backward failed");
    }
    return Tensor(std::vector<size_t>{0});
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

Transformer& Model::block(size_t i) const {
    return *blocks[i];
}

Linear& Model::vocabLinear() const {
    return *vocab;
}

LayerNorm& Model::finalNorm() const {
    return *final_ln;
}

void Model::useStack(StackDevice* slot) {
    stack = slot;
}