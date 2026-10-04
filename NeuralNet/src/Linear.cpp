//
// Created by HP on 10/3/2026.
//

#include "../include/Linear.h"

#include <cmath>

Linear::Linear(size_t input_size, size_t output_size) {
    std::vector<size_t> shape = {output_size, input_size};
    this->weights = Tensor(shape);
    this->weights.randomize(-std::sqrt(1.0f/static_cast<float>(input_size)),
                            std::sqrt(1.0f/static_cast<float>(input_size)));

    this->bias = Tensor({output_size});
    this->bias.fill(0.0f);

    this->weights_grad = Tensor({output_size, input_size}, 0.0f);
    this->bias_grad = Tensor({output_size}, 0.0f);
}

Tensor Linear::forward(const Tensor& input) {
    input_copy = input;
    return input.matmul(weights.transpose()) + bias;
}

Tensor Linear::backward(const Tensor &grad_output) {
    weights_grad = grad_output.transpose().matmul(input_copy);
    bias_grad = grad_output.sum(0);

    return grad_output.matmul(weights);
}
