//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_DROPOUT_H
#define GPT_2_FROM_SCRATCH_DROPOUT_H
#include "Layer.h"

class Dropout : public Layer {
    bool training;
    float rate;

    public:

    Dropout(float rate, bool training);
    Tensor forward(const Tensor& input) override;
};

#endif //GPT_2_FROM_SCRATCH_DROPOUT_H
