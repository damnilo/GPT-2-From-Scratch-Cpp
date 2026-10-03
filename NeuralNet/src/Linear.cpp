//
// Created by HP on 10/3/2026.
//

#include "../include/Linear.h"

#include <cmath>

Linear::Linear(size_t input_size, size_t output_size) {
    std::vector<size_t> shape = {output_size, input_size};
    this->weights = Tensor(shape);
    this->bias = Tensor({output_size});
    this->weights.randomize(-std::sqrt(1.0f/static_cast<float>(input_size)),
                            std::sqrt(1.0f/static_cast<float>(input_size)));
    this->bias.fill(0.0f);
}

Tensor Linear::forward(const Tensor& input) {
    return input.matmul(this->weights.transpose()) + this->bias;
}