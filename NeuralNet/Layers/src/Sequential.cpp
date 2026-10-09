//
// Created by HP on 10/3/2026.
//

#include "../include/Sequential.h"
#include <vector>

void Sequential::addLayer(std::unique_ptr<Layer> layer) {
    this->layers.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor &input) const {
    Tensor output = input;
    for (const auto& l : this->layers) {
        output = l->forward(output);
    }

    return output;
}

Tensor Sequential::backward(const Tensor &grad_output) {
    Tensor output = grad_output;
    for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
        output = (*it)->backward(output);
    }

    return output;
}

std::vector<Tensor *> Sequential::parameters() const {
    std::vector<Tensor*> parameters;

    for (const auto& l : this->layers) {
        for (Tensor* t : l->parameters()) {
            parameters.push_back(t);
        }
    }

    return parameters;
}

std::vector<Tensor *> Sequential::gradients() const {
    std::vector<Tensor*> gradients;

    for (const auto& l : this->layers) {
        for (Tensor* t : l->gradients()) {
            gradients.push_back(t);
        }
    }

    return gradients;
}

void Sequential::zeroGrad() const {
    for (const auto& l : this->layers) {
        l->zeroGrad();
    }
}
