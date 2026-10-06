//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_LINEAR_H
#define GPT_2_FROM_SCRATCH_LINEAR_H
#include "Layer.h"
#include "../../../NumCPP/include/Tensor.h"

class Linear : public Layer {
    Tensor weights;
    Tensor bias;

    Tensor weights_grad;
    Tensor bias_grad;

    Tensor input_copy;

    public:

    Linear(size_t input_size, size_t output_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zeroGrad() override;
};

#endif //GPT_2_FROM_SCRATCH_LINEAR_H
