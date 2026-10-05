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
    input_copy = input;

    auto& shape = input.getShape();
    auto& data = input.getData();

    const size_t batch_size = shape[0];
    const size_t input_size = shape[1];
    const size_t vocab_size = weight.getShape()[0];
    const size_t embedding_dim = weight.getShape()[1];

    Tensor output({batch_size, input_size, embedding_dim}, 0.0f);

    for (size_t i = 0; i < batch_size * input_size; i++) {
        const auto token = static_cast<size_t>(data[i]);

        if (token >= vocab_size) {
            throw std::out_of_range("Token idx out of range");
        }

        for (size_t j = 0; j < weight.getShape()[1]; j++) {
            output[i * embedding_dim + j] = weight[token * embedding_dim + j];
        }
    }

    return output;
}

Tensor Embedding::backward(const Tensor &grad_output) {
    weight_grad.zeros();
    size_t batch_size = grad_output.getShape()[0];
    size_t input_size = grad_output.getShape()[1];
    size_t embedding_dim = grad_output.getShape()[2];

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < input_size; j++) {
            auto id = static_cast<size_t>(input_copy.getData()[i * input_size + j]);

            for (size_t k = 0; k < embedding_dim; k++) {
                weight_grad[id * embedding_dim + k] += grad_output[(input_size * i + j) * embedding_dim + k];
            }
        }
    }

    return weight_grad;
}
