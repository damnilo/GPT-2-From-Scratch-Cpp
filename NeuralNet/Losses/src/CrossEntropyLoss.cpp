//
// Created by HP on 10/5/2026.
//
#include "../include/CrossEntropyLoss.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr float probability_floor = 1e-8f;
}

float CrossEntropyLoss::loss(const Tensor& output, const Tensor& target) {
    const auto& output_data = output.getData();
    const auto& target_data = target.getData();

    if (output_data.size() != target_data.size()) {
        throw std::invalid_argument("CrossEntropyLoss::loss() size mismatch");
    }

    float loss = 0.0f;

    for (size_t i = 0; i < output_data.size(); i++) {
        if (target_data[i] == 0.0f) {
            continue;
        }

        const float probability = std::max(output_data[i], probability_floor);
        loss -= target_data[i] * std::log(probability);
    }

    return loss;
}

Tensor CrossEntropyLoss::backward(const Tensor &output, const Tensor &target) {
    if (output.getShape() != target.getShape()) {
        throw std::invalid_argument("CrossEntropyLoss::backward() size mismatch");
    }

    Tensor grad(output.getShape(), 0.0f);
    const auto& output_data = output.getData();
    const auto& target_data = target.getData();

    for (size_t i = 0; i < output_data.size(); i++) {
        const float probability = std::max(output_data[i], probability_floor);
        grad[i] = -target_data[i] / probability;
    }

    return grad;
}
