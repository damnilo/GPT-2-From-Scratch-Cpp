//
// Created by HP on 10/10/2026.
//

#ifndef GPT_2_FROM_SCRATCH_MODEL_H
#define GPT_2_FROM_SCRATCH_MODEL_H
#include "../../NeuralNet/Layers/include/Sequential.h"
#include "../../NumCPP/include/Tensor.h"

struct ModelConfig {
    size_t vocab_size;
    size_t block_size;
    size_t embedding_dim;
    size_t n_heads;
    size_t n_layers;
    size_t batch_size;
    size_t sequence_len;
};

class Model {

    ModelConfig config;
    Sequential model;

    public:

    Model(size_t vocab_size, size_t block_size, size_t embedding_dim, size_t n_heads, size_t n_layers, size_t batch_size, size_t sequence_len);

    [[nodiscard]] const ModelConfig& getConfig() const;

    [[nodiscard]] Tensor forward(const Tensor &input) const;
    Tensor backward(const Tensor &grad_output);

    [[nodiscard]] std::vector<Tensor*> parameters() const;
    [[nodiscard]] std::vector<Tensor*> gradients() const;

    void zeroGrad() const;
};

#endif //GPT_2_FROM_SCRATCH_MODEL_H
