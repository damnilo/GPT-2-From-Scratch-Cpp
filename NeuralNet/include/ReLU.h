//
// Created by HP on 10/3/2026.
//

#ifndef GPT_2_FROM_SCRATCH_RELU_H
#define GPT_2_FROM_SCRATCH_RELU_H
#include "Layer.h"

class ReLU : public Layer {

    public:

    Tensor forward(const Tensor& input) override;
};

#endif //GPT_2_FROM_SCRATCH_RELU_H
