//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_LAYER_H
#define GPT_2_FROM_SCRATCH_LAYER_H
#include "../../../NumCPP/include/Tensor.h"

class Layer {

    public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;

    virtual std::vector<Tensor*> parameters() {
        return {};
    }

    virtual std::vector<Tensor*> gradients() {
        return {};
    }

    virtual void zeroGrad() {}
};

#endif //GPT_2_FROM_SCRATCH_LAYER_H
