//
// Created by HP on 10/3/2026.
//
#include "../include/Dropout.h"

#include <random>
#include <stdexcept>

Dropout::Dropout(float rate, bool training) {
    if (rate < 0.0f || rate > 1.0f) {
        throw std::invalid_argument("rate cannot be lower than 0.0f nor higher than 1.0f");
    }
    this->rate = rate;
    this->training = training;
}

Tensor Dropout::forward(const Tensor& input) {
    if (!training) {
        return input;
    }

    if (rate == 1.0f) {
        return Tensor(input.getShape());
    }

    if (rate == 0.0f) return input;

    Tensor mask(input.getShape());
    std::vector<float> output(input.getShape().size());

    mask.randomize(0.0f, 1.0f);

    for (int i = 0; i < input.getData().size(); i++) {
        if (mask.getData()[i] <= rate) {
            output[i] = 0.0f;
        }
    }

    return {input.getShape(), output};
}