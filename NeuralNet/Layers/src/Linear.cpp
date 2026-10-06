//
// Created by HP on 10/3/2026.
//

#include "../include/Linear.h"

#include <cmath>
#include <stdexcept>

namespace {

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
