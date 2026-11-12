//
// Created by HP on 10/3/2026.
//

#include "../include/Linear.h"
#include "../../../Cuda/include/Vocab.h"

#include <cmath>
#include <stdexcept>

namespace {

void writeGradient(Tensor& destination, const std::vector<float>& source) {
    if (destination.size() != source.size()) {
        throw std::runtime_error("gradient size mismatch");
    }
    destination = Tensor(destination.getShape(), source);
}

Tensor flattenLeading(const Tensor& input) {
    const auto& shape = input.getShape();

    if (shape.empty()) {
        throw std::invalid_argument("Linear received an empty input");
    }

    const size_t features = shape.back();

    if (features == 0 || input.size() % features != 0) {
        throw std::invalid_argument("Linear feature size does not match the input");
    }

    return input.reshape({input.size() / features, features});
}

}

Linear::Linear(size_t input_size, size_t output_size) {
    std::vector<size_t> shape = {output_size, input_size};
    this->weights = Tensor(shape);
    this->weights.randomize(-std::sqrt(1.0f/static_cast<float>(input_size)),
                            std::sqrt(1.0f/static_cast<float>(input_size)));

    this->bias = Tensor({output_size});
    this->bias.fill(0.0f);

    this->weights_grad = Tensor({output_size, input_size}, 0.0f);
    this->bias_grad = Tensor({output_size}, 0.0f);
}

Tensor Linear::forward(const Tensor& input) {
    if (device != nullptr) {
        if (input.ndim() != 3 || input.getShape()[0] != 1 || input.getShape().back() != weights.getShape()[1]) {
            throw std::invalid_argument("Linear::forward: device projection needs batch 1");
        }
        input_copy = input;
        const size_t T = input.getShape()[1];
        std::vector<float> host_logits;
        if (!forwardVocabDevice(device->path, *this, input.getData().data(), T, device->state, host_logits)) {
            throw std::runtime_error("vocabulary forward on device failed");
        }
        std::vector<size_t> out_shape = input.getShape();
        out_shape.back() = weights.getShape()[0];
        return Tensor(out_shape, host_logits);
    }

    if (input.getShape().empty() || input.getShape().back() != weights.getShape()[1]) {
        throw std::invalid_argument("Linear::forward: feature size does not match weights");
    }

    input_copy = input;
    Tensor flat = flattenLeading(input);
    Tensor out = flat.matmul(weights.transpose()) + bias;

    std::vector<size_t> out_shape = input.getShape();
    out_shape.back() = weights.getShape()[0];
    return out.reshape(out_shape);
}

Tensor Linear::backward(const Tensor &grad_output) {
    if (device != nullptr) {
        if (grad_output.ndim() != 3 || grad_output.getShape()[0] != 1 || grad_output.getShape().back() != weights.getShape()[0]) {
            throw std::invalid_argument("Linear::backward: device gradient needs batch 1");
        }
        VocabGrad grad;
        if (!backwardVocabDevice(device->path, device->state, *this, grad_output.getData().data(), &grad)) {
            throw std::runtime_error("vocabulary backward on device failed");
        }
        writeGradient(weights_grad, grad.W);
        writeGradient(bias_grad, grad.b);
        std::vector<size_t> in_shape = grad_output.getShape();
        in_shape.back() = weights.getShape()[1];
        return Tensor(in_shape, grad.input);
    }

    if (input_copy.getShape().empty() || grad_output.getShape().empty() ||
        grad_output.getShape().back() != weights.getShape()[0] ||
        grad_output.size() / weights.getShape()[0] != input_copy.size() / weights.getShape()[1]) {
        throw std::invalid_argument("Linear::backward: gradient shape does not match the forward input");
    }

    Tensor grad_flat = flattenLeading(grad_output);
    Tensor input_flat = flattenLeading(input_copy);

    weights_grad = grad_flat.transpose().matmul(input_flat);
    bias_grad = grad_flat.sum(0);

    return grad_flat.matmul(weights).reshape(input_copy.getShape());
}

std::vector<Tensor*> Linear::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> Linear::gradients() {
    return {&weights_grad, &bias_grad};
}

void Linear::zeroGrad() {
    weights_grad.zeros();
    bias_grad.zeros();
}

void Linear::useDevice(VocabDevice* slot) {
    device = slot;
}
