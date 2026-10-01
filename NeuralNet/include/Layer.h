//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_LAYER_H
#define GPT_2_FROM_SCRATCH_LAYER_H
#include "../../NumCPP/include/Tensor.h"

class Layer {
    public:

    virtual Tensor forward(const Tensor& input) = 0;

    virtual ~Layer() = default;
};

#endif //GPT_2_FROM_SCRATCH_LAYER_H
