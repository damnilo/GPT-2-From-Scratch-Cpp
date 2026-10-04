//
// Created by HP on 10/3/2026.
//

#ifndef GPT_2_FROM_SCRATCH_RELU_H
#define GPT_2_FROM_SCRATCH_RELU_H
#include "Layer.h"

class ReLU : public Layer {
    std::vector<int> mask;

    public:

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};

#endif //GPT_2_FROM_SCRATCH_RELU_H
