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
    std::vector<float> data = input.getData();
    std::vector<size_t> shape = input.getShape();

    size_t last_row = shape.back();

    if (last_row != this->gamma.getShape()[0]) {
        throw std::invalid_argument("LayerNorm::forward: normalized state does not match number of dimensions");
    }

    size_t num_vectors = data.size() / last_row;

    Tensor output(shape, 0.0f);

    for (size_t i = 0; i < num_vectors; i++) {
        size_t offset = i * last_row;

        float mean = 0.0f;

        for (size_t j = 0; j < last_row; j++) {
            mean += data[offset + j];
        }

        mean /= static_cast<float>(last_row);

        float variance = 0.0f;
        for (size_t j = 0; j < last_row; j++) {
            float diff = data[offset + j] - mean;
            variance += diff * diff;
        }

        variance /= static_cast<float>(last_row);

        float denominator = std::sqrt(variance + eps);

        for (size_t j = 0; j < last_row; j++) {
            float norm = (data[offset + j] - mean) / denominator;

            output[offset + j] = gamma[j] * norm + beta[j];
        }
    }

    return output;
}