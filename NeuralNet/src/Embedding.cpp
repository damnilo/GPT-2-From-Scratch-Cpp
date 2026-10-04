//
// Created by HP on 10/3/2026.
//

#include "../include/Embedding.h"

#include <stdexcept>

Embedding::Embedding(size_t input_size, size_t output_size) {
    std::vector<size_t> shape = {input_size, output_size};
    this->weight = Tensor(shape);
    this->weight.randomize();
}

Tensor Embedding::forward(const Tensor& input) {
    auto& shape = input.getShape();
    auto& data = input.getData();

    const size_t batch_size = shape[0];
    const size_t input_size = shape[1];
    const size_t vocab_size = weight.getShape()[0];
    const size_t embedding_dim = weight.getShape()[1];

    Tensor output({batch_size, input_size, embedding_dim}, 0.0f);

    for (size_t i = 0; i < batch_size * input_size; i++) {
        const auto token = static_cast<size_t>(data[i]);

        if (token > vocab_size) {
            throw std::out_of_range("Token idx out of range");
        }

        for (size_t j = 0; j < weight.getShape()[1]; j++) {
            output[i * embedding_dim + j] = weight[token * embedding_dim + j];
        }
    }

    return output;
}

Tensor Embedding::backward(const Tensor &grad_output) {
}
