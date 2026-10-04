//
// Created by HP on 10/3/2026.
//

#include "../include/Sequential.h"
#include <vector>

void Sequential::addLayer(std::unique_ptr<Layer> layer) {
    this->layers.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor &input) {
    Tensor output = input;
    for (const auto& l : this->layers) {
        output = l->forward(output);
    }

    return output;
}

Tensor Sequential::backward(const Tensor &grad_output) {
    Tensor output = grad_output;
    for (const auto& l : this->layers) {
        output = l->backward(output);
    }

    return output;
}