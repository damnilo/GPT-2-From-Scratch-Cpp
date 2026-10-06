//
// Created by HP on 10/5/2026.
//
#include "../include/AdamW.h"

#include <cmath>
#include <stdexcept>

AdamW::AdamW(float learning_rate, float weight_decay, float beta1, float beta2, float epsilon) {
    this->learning_rate = learning_rate;
    this->weight_decay = weight_decay;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;

    this->timestep = 0;
}

void AdamW::step(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& grad) {
    if (parameters.size() != grad.size()) {
        throw std::invalid_argument("parameter size does not match gradient");
    }

    if (m.empty()) {
        for (Tensor* p : parameters) {
            m.emplace_back(p->getShape(), 0.0f);
            v.emplace_back(p->getShape(), 0.0f);
        }
    }

    ++timestep;
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));

    for (size_t i = 0; i < parameters.size(); i++) {
        Tensor& param = *parameters[i];
        const Tensor& g = *grad[i];

        param = param * (1.0f - learning_rate * weight_decay);
        m[i] = m[i] * beta1 + g * (1.0f - beta1);
        v[i] = v[i] * beta2 + g.pow(2) * (1.0f - beta2);

        Tensor m_hat = m[i] / bias_correction1;
        Tensor v_hat = v[i] / bias_correction2;

        param = param - (m_hat * learning_rate) / (v_hat.sqrt() + epsilon);
    }
}

size_t AdamW::getTimestep() const {
    return timestep;
}
