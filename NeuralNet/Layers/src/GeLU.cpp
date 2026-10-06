//
// Created by HP on 10/3/2026.
//

#include "../include/GeLU.h"
#include "../../NumCPP/include/Math.h"
#include <valarray>

float GeLU::gelu(float x) const {
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
}

Tensor GeLU::forward(const Tensor& input) {
    const std::vector<float>& data = input.getData();
    input_copy = input.getData();
    std::vector<float> ret(data.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        float x = data[i];
        ret[i] = gelu(x);
    }

    return {input.getShape(), ret};
}

Tensor GeLU::backward(const Tensor &grad_output) {
    const std::vector<float>& data = grad_output.getData();
    std::vector<float> ret(data.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        float x = input_copy[i];
        ret[i] = data[i] * Math::derivative(x, this, &GeLU::gelu);
    }

    return {grad_output.getShape(), ret};
}
