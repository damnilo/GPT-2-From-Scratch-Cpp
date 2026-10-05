//
// Created by HP on 10/5/2026.
//
#include "../include/AdamW.h"
#include "../../NumCPP/include/Math.h"

AdamW::AdamW(float learning_rate, float weight_decay, float beta1, float beta2, float epsilon) {
    this->learning_rate = learning_rate;
    this->weight_decay = weight_decay;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;

    this->timestep = 0;
}

void AdamW::step(std::vector<Tensor> &parameters, const std::vector<Tensor> &grad) {
    if (parameters.size() != grad.size()) {
        throw std::invalid_argument("parameter size does not match gradient");
    }

    if (m.empty()) {
        for (Tensor& p : parameters) {
            m.emplace_back(p.getShape(), 0.0f);
            v.emplace_back(p.getShape(), 0.0f);
        }
    }
    std::vector<Tensor> m_hat(m.size());
    std::vector<Tensor> v_hat(v.size());

    ++timestep;
    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i] = parameters[i] * (1 - learning_rate * weight_decay);
        m[i] = m[i] * beta1 + grad[i] * (1 - beta1);
        v[i] = v[i] * beta2 + grad[i].pow(2) * (1 - beta2);

        m_hat[i] = m[i] / static_cast<float>(1.0f - std::pow(beta1, timestep));
        v_hat[i] = v[i] / static_cast<float>(1.0f - std::pow(beta2, timestep));

        parameters[i] = parameters[i] - (m_hat[i] * learning_rate) / (v_hat[i].sqrt() + epsilon);
    }
}

size_t AdamW::getTimestep() const {
    return timestep;
}
