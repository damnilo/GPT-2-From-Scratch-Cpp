//
// Created by HP on 10/4/2026.
//

#ifndef GPT_2_FROM_SCRATCH_SOFTMAX_H
#define GPT_2_FROM_SCRATCH_SOFTMAX_H
#include "Layer.h"

class Softmax : public Layer {
    int axis;

    public:

    Softmax(int axis);
    Tensor forward(const Tensor& input) override;
};

#endif //GPT_2_FROM_SCRATCH_SOFTMAX_H
