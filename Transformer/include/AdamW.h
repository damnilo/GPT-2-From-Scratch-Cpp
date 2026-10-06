//
// Created by HP on 10/5/2026.
//

#ifndef GPT_2_FROM_SCRATCH_ADAMW_H
#define GPT_2_FROM_SCRATCH_ADAMW_H
#include "../../NumCPP/include/Tensor.h"

class AdamW {
    float learning_rate;
    float weight_decay;
    float beta1;
    float beta2;
    float epsilon;

    size_t timestep;

    std::vector<Tensor> m;
    std::vector<Tensor> v;

    public:

    AdamW(float learning_rate = 1e-3, float weight_decay = 1e-2, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8);
    void step(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& grad);
    [[nodiscard]] size_t getTimestep() const;
};

#endif //GPT_2_FROM_SCRATCH_ADAMW_H
