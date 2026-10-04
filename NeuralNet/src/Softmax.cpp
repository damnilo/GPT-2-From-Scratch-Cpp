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
    const auto& data = input.getData();

    if (shape.empty()) {
        throw std::invalid_argument("Softmax cannot be applied to a scalar.");
    }

    if (data.empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }

    if (axis == -1) {
        axis = static_cast<int>(shape.size()) - 1;
    }

    if (axis < 0 || axis >= shape.size()) {
        throw std::invalid_argument("The axis must be greater than 0 or -1.");
    }

    Tensor ret(shape, 0.0f);

    size_t axis_size = shape[axis];
    size_t axis_stride = input.getStrides()[axis];

    size_t out_size = 1;

    for (int i = 0; i < axis; i++) {
        out_size *= shape[i];
    }

    size_t inner_size = axis_stride;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(out_size); i++) {
        for (size_t k = 0; k < inner_size; k++) {
            size_t group_start = i * axis_size * inner_size + k;
            float max_value = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < axis_size; j++) {
                size_t idx = group_start + j * axis_stride;
                max_value = std::max(max_value, data[idx]);
            }

            float sum = 0.0f;

            for (size_t j = 0; j < axis_size; j++) {
                size_t idx = group_start + j * axis_stride;

                ret[idx] = std::exp(data[idx] - max_value);
                sum += ret[idx];
            }

            for (size_t j = 0; j < axis_size; j++) {
                size_t idx = group_start + j * axis_stride;

                ret[idx] /= sum;
            }
        }
    }

    return ret;
}