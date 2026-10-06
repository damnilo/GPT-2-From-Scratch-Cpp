//
// Created by HP on 10/3/2026.
//
#include "../include/ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    const auto& data = input.getData();
    std::vector<float> ret(data.size());
    mask.resize(data.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        ret[i] = std::max(0.0f, data[i]);
        if (ret[i] > 0.0) mask[i] = 1;
        else mask[i] = 0;
    }

    return {input.getShape(), ret};
}

Tensor ReLU::backward(const Tensor &grad_output) {
    const auto& data = grad_output.getData();
    std::vector<float> ret(data.size());
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        ret[i] = data[i] * static_cast<float>(mask[i]);
    }

    return {grad_output.getShape(), ret};
}
