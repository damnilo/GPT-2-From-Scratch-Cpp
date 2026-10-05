//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_EMBEDDING_H
#define GPT_2_FROM_SCRATCH_EMBEDDING_H

#include "Layer.h"
#include "../../NumCPP/include/Tensor.h"

class Embedding : public Layer{
    Tensor weight;
    Tensor weight_grad;

    Tensor input_copy;

    public:
    Embedding(size_t input_size, size_t output_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};

#endif //GPT_2_FROM_SCRATCH_EMBEDDING_H
