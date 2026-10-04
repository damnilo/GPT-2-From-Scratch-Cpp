//
// Created by HP on 10/3/2026.
//

#include "../include/GeLU.h"

#include <valarray>

Tensor GeLU::forward(const Tensor& input) {
    const std::vector<float>& data = input.getData();
    std::vector<float> ret(data.size());

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        float x = data[i];
        ret[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
    }

    return {input.getShape(), ret};
}

Tensor GeLU::backward(const Tensor &grad_output) {
}
