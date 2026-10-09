//
// Created by HP on 10/9/2026.
//

#ifndef GPT_2_FROM_SCRATCH_FEEDFORWARD_H
#define GPT_2_FROM_SCRATCH_FEEDFORWARD_H
#include "../../NeuralNet/Layers/include/Layer.h"
#include "../../NeuralNet/Layers/include/Sequential.h"

class FeedForward : public Layer{

    Sequential s;

    public:

    FeedForward(size_t input, size_t output);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor &grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    void zeroGrad() override;
};

#endif //GPT_2_FROM_SCRATCH_FEEDFORWARD_H
