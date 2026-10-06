//
// Created by HP on 10/4/2026.
//

#include "../include/Softmax.h"

#include <stdexcept>

Softmax::Softmax(int axis) {
    this->axis = axis;
}

int Softmax::resolvedAxis(const std::vector<size_t>& shape) const {
    int actual = axis;

    if (actual == -1) {
        actual = static_cast<int>(shape.size()) - 1;
    }

    if (actual < 0 || static_cast<size_t>(actual) >= shape.size()) {
        throw std::invalid_argument("The axis must be greater than 0 or -1.");
    }

    return actual;
}

Tensor Softmax::forward(const Tensor& input) {
    const auto& shape = input.getShape();
    if (shape.empty()) {
        throw std::invalid_argument("Softmax cannot be applied to a scalar.");
    }

    if (input.getData().empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }

    const int actual_axis = resolvedAxis(shape);
    Tensor max_values = input.max(actual_axis, true);
    Tensor shifted = input - max_values;
    Tensor exp_values = shifted.exp();
    Tensor sums = exp_values.sum(actual_axis, true);

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

    if (grad_output.getShape() != output.getShape()) {
        throw std::invalid_argument("Softmax::backward: gradient shape does not match the forward output");
    }

    const int actual_axis = resolvedAxis(grad_output.getShape());
    Tensor weighted = grad_output * output;
    Tensor sum = weighted.sum(actual_axis, true);

    return output * (grad_output - sum);
}
