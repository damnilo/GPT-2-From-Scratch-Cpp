//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_LAYERNORM_H
#define GPT_2_FROM_SCRATCH_LAYERNORM_H
#include "Layer.h"
#include "../../NumCPP/include/Tensor.h"

class LayerNorm : public Layer {
    Tensor beta;
    Tensor gamma;
    float eps = 1e-5;

    void betaInit();
    void gammaInit();
public:

    LayerNorm(size_t normalized_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};

#endif //GPT_2_FROM_SCRATCH_LAYERNORM_H
