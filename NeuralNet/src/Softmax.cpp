//
// Created by HP on 10/4/2026.
//

#include "../include/Softmax.h"

#include <cmath>
#include <stdexcept>

Softmax::Softmax(int axis) {
    this->axis = axis;
}

Tensor Softmax::forward(const Tensor& input) {
    const auto& shape = input.getShape();
    if (shape.empty()) {
        throw std::invalid_argument("Softmax cannot be applied to a scalar.");
    }

    if (input.getData().empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }
    int actual_axis = axis;

    if (axis == -1) {
        axis = static_cast<int>(shape.size()) - 1;
    }

    if (axis < 0 || axis >= shape.size()) {
        throw std::invalid_argument("The axis must be greater than 0 or -1.");
    }

    Tensor max_values = input.max(actual_axis);
    Tensor shifted = input - max_values;
    Tensor exp_values = shifted.exp();
    Tensor sums = exp_values.sum(actual_axis);

    output = exp_values / sums;
    return output;
}

Tensor Softmax::backward(const Tensor &grad_output) {

    if (grad_output.getData().empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }

    if (grad_output.getShape().empty()) {
        throw std::invalid_argument("The shape cannot be empty.");
    }

    Tensor weighted = grad_output * output;
    Tensor sum = weighted.sum(axis);

    return output * (grad_output - sum);
}
