//
// Created by HP on 10/3/2026.
//

#include "../include/Embedding.h"

#include <stdexcept>

Embedding::Embedding(size_t input_size, size_t output_size) {
    std::vector<size_t> shape = {input_size, output_size};
    this->weight = Tensor(shape);
    this->weight.randomize(-0.02f, 0.02f);
    this->weight_grad = Tensor(shape, 0.0f);
}

Tensor Embedding::forward(const Tensor& input) {
    const auto& shape = input.getShape();
    const auto& data = input.getData();

    if (shape.empty() || shape.size() > 2) {
        throw std::invalid_argument("Embedding input must have rank 1 or 2");
    }

    input_copy = input;

    const size_t batch_size = shape.size() == 2 ? shape[0] : 1;
    const size_t input_size = shape.size() == 2 ? shape[1] : shape[0];
    const size_t vocab_size = weight.getShape()[0];
    const size_t embedding_dim = weight.getShape()[1];

    const std::vector<size_t> out_shape = shape.size() == 2
        ? std::vector<size_t>{batch_size, input_size, embedding_dim}
        : std::vector<size_t>{input_size, embedding_dim};

    Tensor output(out_shape, 0.0f);

    for (size_t i = 0; i < batch_size * input_size; i++) {
        const float raw = data[i];

        if (raw < 0.0f) {
            throw std::out_of_range("Token idx out of range");
        }

        const auto token = static_cast<size_t>(raw);

        if (token >= vocab_size) {
            throw std::out_of_range("Token idx out of range");
        }

        for (size_t j = 0; j < embedding_dim; j++) {
            output[i * embedding_dim + j] = weight[token * embedding_dim + j];
        }
    }

    return output;
}

Tensor Embedding::backward(const Tensor &grad_output) {
    const size_t embedding_dim = weight.getShape()[1];

    if (weight_grad.getShape() != weight.getShape()) {
        weight_grad = Tensor(weight.getShape(), 0.0f);
    } else {
        weight_grad.zeros();
    }

    if (grad_output.getShape().empty() || grad_output.getShape().back() != embedding_dim) {
        throw std::invalid_argument("Embedding::backward: gradient shape mismatch");
    }

    const size_t tokens = grad_output.size() / embedding_dim;
    const auto& ids = input_copy.getData();

    if (ids.size() != tokens) {
        throw std::invalid_argument("Embedding::backward: token count mismatch");
    }

    const auto& grad_data = grad_output.getData();

    for (size_t i = 0; i < tokens; i++) {
        const float raw = ids[i];

        if (raw < 0.0f) {
            throw std::out_of_range("Token idx out of range");
        }

        const auto id = static_cast<size_t>(raw);

        for (size_t k = 0; k < embedding_dim; k++) {
            weight_grad[id * embedding_dim + k] += grad_data[i * embedding_dim + k];
        }
    }

    return Tensor();
}

std::vector<Tensor*> Embedding::parameters() {
    return {&weight};
}

std::vector<Tensor*> Embedding::gradients() {
    return {&weight_grad};
}

void Embedding::zeroGrad() {
    weight_grad.zeros();
}
