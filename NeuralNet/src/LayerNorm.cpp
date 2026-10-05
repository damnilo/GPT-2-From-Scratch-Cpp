//
// Created by HP on 10/3/2026.
//
#include "../include/LayerNorm.h"
#include <cmath>
#include <stdexcept>

void LayerNorm::betaInit() {
    this->beta.zeros();
}

void LayerNorm::gammaInit() {
    this->gamma.ones();
}

LayerNorm::LayerNorm(size_t last_row) {
    this->gamma = Tensor({last_row});
    this->beta = Tensor({last_row});

    betaInit();
    gammaInit();
}

Tensor LayerNorm::forward(const Tensor& input) {
    const std::vector<float>& data = input.getData();
    const std::vector<size_t>& shape = input.getShape();

    if (shape.empty()) {
        throw std::runtime_error("LayerNorm::forward: empty input");
    }

    size_t last_row = shape.back();

    if (last_row != this->gamma.getShape()[0]) {
        throw std::invalid_argument("LayerNorm::forward: normalized state does not match number of dimensions");
    }

    input_copy = input;

    size_t num_vectors = data.size() / last_row;

    Tensor output(shape, 0.0f);
    normalized = Tensor(shape, 0.0f);
    variance = Tensor({num_vectors}, 0.0f);
    for (size_t i = 0; i < num_vectors; i++) {
        size_t offset = i * last_row;

        float mean = 0.0f;

        for (size_t j = 0; j < last_row; j++) {
            mean += data[offset + j];
        }

        mean /= static_cast<float>(last_row);

        float var = 0.0f;
        for (size_t j = 0; j < last_row; j++) {
            float diff = data[offset + j] - mean;
            var += diff * diff;
        }

        var /= static_cast<float>(last_row);
        variance[i] = var;
        float denominator = std::sqrt(var + eps);

        for (size_t j = 0; j < last_row; j++) {
            float norm = (data[offset + j] - mean) / denominator;
            normalized[offset + j] = norm;

            output[offset + j] = gamma[j] * norm + beta[j];
        }
    }

    return output;
}

Tensor LayerNorm::backward(const Tensor &grad_output) {
    const auto& shape = grad_output.getShape();

    if (grad_output.getShape() != shape) {
        throw std::invalid_argument("LayerNorm::backward: input shape mismatch");
    }

    size_t last_row = shape.back();
    size_t num_vectors = input_copy.size() / last_row;

    beta_grad = grad_output.sum(0);

    while (beta_grad.ndim() > 1) {
        beta_grad = beta_grad.sum(0);
    }

    gamma_grad = grad_output * normalized;
    gamma_grad = gamma_grad.sum(0);

    while (gamma_grad.ndim() > 1) {
        gamma_grad = gamma_grad.sum(0);
    }

    Tensor grad_input(shape, 0.0f);

    const auto& grad_data = grad_output.getData();
    const auto& norm_data = normalized.getData();
    const auto& gamma_data = gamma_grad.getData();
    const auto& var_data = variance.getData();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_vectors); i++) {
        size_t offset = static_cast<size_t>(i) * last_row;
        float sum_grad = 0.0f;
        float sum_grad_norm = 0.0f;

        for (size_t j = 0; j < last_row; j++) {
            size_t idx = offset + j;
            sum_grad += grad_data[idx];
            sum_grad_norm += norm_data[idx] * grad_data[idx];
        }

        float inv_std = 1.0f / std::sqrt(var_data[i] + eps);
        auto hidden_size = static_cast<float>(last_row);

        for (size_t j = 0; j < last_row; j++) {
            size_t idx = offset + j;

            grad_input[idx] = gamma_data[j] * inv_std * (grad_data[idx] - sum_grad / hidden_size -
                                norm_data[idx] * sum_grad_norm / hidden_size);
        }
    }

    return grad_input;
}
