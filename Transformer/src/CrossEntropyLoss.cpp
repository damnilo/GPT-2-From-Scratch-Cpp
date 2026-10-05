//
// Created by HP on 10/5/2026.
//
#include "../include/CrossEntropyLoss.h"

#include <valarray>

float CrossEntropyLoss::loss(const Tensor& output, const Tensor& target) {
    float loss = 0.0f;

    const auto& output_data = output.getData();
    const auto& target_data = target.getData();

    if (output_data.size() != target_data.size()) {
        throw std::invalid_argument("CrossEntropyLoss::loss() size mismatch");
    }

    for (int i = 0; i < output_data.size(); i++) {
        loss -= target_data[i] * std::log(output_data[i]);
    }

    return loss;
}

Tensor CrossEntropyLoss::backward(const Tensor &output, const Tensor &target) {
    if (output.getShape() != target.getShape()) {
        throw std::invalid_argument("CrossEntropyLoss::backward() size mismatch");
    }

    return output - target;
}
