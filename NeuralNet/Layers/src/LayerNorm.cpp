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
    this->gamma_grad = Tensor({last_row}, 0.0f);
    this->beta_grad = Tensor({last_row}, 0.0f);

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
    const auto& shape = input_copy.getShape();

    if (grad_output.getShape() != shape || normalized.getShape() != shape) {
        throw std::invalid_argument("LayerNorm::backward: input shape mismatch");
    }

    size_t last_row = shape.back();
    size_t num_vectors = input_copy.size() / last_row;

    beta_grad = Tensor({last_row}, 0.0f);
    gamma_grad = Tensor({last_row}, 0.0f);

    const auto& grad_data = grad_output.getData();
    const auto& norm_data = normalized.getData();
    const auto& gamma_data = gamma.getData();
    const auto& var_data = variance.getData();

    for (size_t i = 0; i < num_vectors; i++) {
        const size_t offset = i * last_row;

        for (size_t j = 0; j < last_row; j++) {
            beta_grad[j] += grad_data[offset + j];
            gamma_grad[j] += grad_data[offset + j] * norm_data[offset + j];
        }
    }

    Tensor grad_input(shape, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_vectors); i++) {
        const size_t offset = static_cast<size_t>(i) * last_row;
        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;

        for (size_t j = 0; j < last_row; j++) {
            const size_t idx = offset + j;
            const float dxhat = grad_data[idx] * gamma_data[j];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * norm_data[idx];
        }

        const float inv_std = 1.0f / std::sqrt(var_data[static_cast<size_t>(i)] + eps);
        const float inv_n = 1.0f / static_cast<float>(last_row);

        for (size_t j = 0; j < last_row; j++) {
            const size_t idx = offset + j;
            const float dxhat = grad_data[idx] * gamma_data[j];
            grad_input[idx] = inv_std * (dxhat - sum_dxhat * inv_n - norm_data[idx] * sum_dxhat_xhat * inv_n);
        }
    }

    return grad_input;
}

std::vector<Tensor*> LayerNorm::parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> LayerNorm::gradients() {
    return {&gamma_grad, &beta_grad};
}

void LayerNorm::zeroGrad() {
    gamma_grad.zeros();
    beta_grad.zeros();
}
