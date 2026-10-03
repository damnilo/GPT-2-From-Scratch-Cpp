//
// Created by HP on 10/3/2026.
//

#ifndef GPT_2_FROM_SCRATCH_GELU_H
#define GPT_2_FROM_SCRATCH_GELU_H
#include "Layer.h"

class GeLU : public Layer {
    float coeff = 0.044715f;
    float sqrt_2_over_pi = 0.7978845608f;

    public:

    Tensor forward(const Tensor& input) override;
};

#endif //GPT_2_FROM_SCRATCH_GELU_H
