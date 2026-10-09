//
// Created by HP on 10/9/2026.
//
#include "../include/FeedForward.h"

#include "../../NeuralNet/Layers/include/GeLU.h"
#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Layers/include/ReLU.h"


FeedForward::FeedForward(size_t input, size_t output) {
    s.addLayer(std::make_unique<Linear>(input, 4 * input));
    s.addLayer(std::make_unique<GeLU>());
    s.addLayer(std::make_unique<Linear>(4 * input, output));
}

Tensor FeedForward::forward(const Tensor &input) {
    return s.forward(input);
}

Tensor FeedForward::backward(const Tensor &grad_output) {
    return s.backward(grad_output);
}

std::vector<Tensor *> FeedForward::parameters() {
    return s.parameters();
}

std::vector<Tensor *> FeedForward::gradients() {
    return s.gradients();
}

void FeedForward::zeroGrad() {
    s.zeroGrad();
}
