//
// Created by HP on 10/9/2026.
//
#include "../include/FeedForward.h"
#include "../../Cuda/include/FeedForward.h"

#include "../../NeuralNet/Layers/include/GeLU.h"
#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Layers/include/ReLU.h"

#include <stdexcept>

namespace {

void writeGradient(Tensor& destination, const std::vector<float>& source) {
    if (destination.size() != source.size()) {
        throw std::runtime_error("gradient size mismatch");
    }
    destination = Tensor(destination.getShape(), source);
}

}


FeedForward::FeedForward(size_t input, size_t output) {
    s.addLayer(std::make_unique<Linear>(input, 4 * input));
    s.addLayer(std::make_unique<GeLU>());
    s.addLayer(std::make_unique<Linear>(4 * input, output));
}

Tensor FeedForward::forward(const Tensor &input) {
    if (device == nullptr) {
        return s.forward(input);
    }
    if (input.ndim() != 3 || input.getShape()[0] != 1) {
        throw std::runtime_error("device feed-forward needs batch 1");
    }

    const size_t T = input.getShape()[1];
    std::vector<float> host_out;
    if (!forwardFeedForwardDevice(device->path, *this, input.getData().data(), T, device->state, host_out)) {
        throw std::runtime_error("feed-forward forward on device failed");
    }
    return Tensor(input.getShape(), host_out);
}

Tensor FeedForward::backward(const Tensor &grad_output) {
    if (device == nullptr) {
        return s.backward(grad_output);
    }

    FeedForwardGrad grad;
    if (!backwardFeedForwardDevice(device->path, device->state, *this, grad_output.getData().data(), grad)) {
        throw std::runtime_error("feed-forward backward on device failed");
    }

    const std::vector<Tensor*> grads = gradients();
    if (grads.size() != 4) {
        throw std::runtime_error("feed-forward gradient layout mismatch");
    }
    writeGradient(*grads[0], grad.W1);
    writeGradient(*grads[1], grad.b1);
    writeGradient(*grads[2], grad.W2);
    writeGradient(*grads[3], grad.b2);
    return Tensor(grad_output.getShape(), grad.input);
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

void FeedForward::useDevice(FeedForwardDevice* slot) {
    device = slot;
}
