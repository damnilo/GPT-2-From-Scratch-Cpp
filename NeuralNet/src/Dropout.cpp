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

    Tensor output(input.getShape(), 0.0f);
    mask = {input.getShape(), 0.0f};

    mask.randomize(0.0f, 1.0f);

    const auto& input_data = input.getData();
    const auto& mask_data = mask.getData();

    for (int i = 0; i < input.getData().size(); i++) {
        if (mask_data[i] <= rate) {
            output[i] = 0.0f;
            mask[i] = 1.0f;
        }else {
            output[i] = input_data[i] / (1.0f - rate);
            mask[i] = 0.0f;
        }
    }

    return output;
}

Tensor Dropout::backward(const Tensor &grad_output) {
    Tensor ret(grad_output.getShape(), 0.0f);

    const auto& grad_data = grad_output.getData();

    for (int i = 0; i < grad_output.getData().size(); i++) {
        ret[i] = grad_data[i] * mask[i] / (1.0f - rate);
    }

    return ret;
}
