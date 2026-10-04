//
// Created by HP on 10/3/2026.
//
#include "../include/ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    const auto& data = input.getData();
    std::vector<float> ret(data.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        ret[i] = std::max(0.0f, data[i]);
    }

    return {input.getShape(), ret};
}

Tensor ReLU::backward(const Tensor &grad_output) {
}
