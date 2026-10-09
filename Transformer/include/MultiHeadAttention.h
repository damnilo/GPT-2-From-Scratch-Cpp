//
// Created by HP on 10/9/2026.
//

#ifndef GPT_2_FROM_SCRATCH_MULTIHEADATTENTION_H
#define GPT_2_FROM_SCRATCH_MULTIHEADATTENTION_H
#include "Attention.h"

class MultiHeadAttention : public Layer {

    size_t head_dim;
    Linear out_linear;
    std::vector<Attention> heads;

    public:

    MultiHeadAttention(size_t n_heads, size_t batch_size, size_t sequence_len, size_t embedding_dim);

    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zeroGrad() override;
};

#endif //GPT_2_FROM_SCRATCH_MULTIHEADATTENTION_H
