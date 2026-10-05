//
// Created by HP on 10/5/2026.
//

#ifndef GPT_2_FROM_SCRATCH_CROSSENTROPYLOSS_H
#define GPT_2_FROM_SCRATCH_CROSSENTROPYLOSS_H
#include "../../NumCPP/include/Tensor.h"

class CrossEntropyLoss {

    public:

    CrossEntropyLoss() = default;
    static float loss(const Tensor& output, const Tensor& target);
    static Tensor backward(const Tensor& output, const Tensor& target);
};

#endif //GPT_2_FROM_SCRATCH_CROSSENTROPYLOSS_H
