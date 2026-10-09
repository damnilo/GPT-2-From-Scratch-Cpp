//
// Created by HP on 10/9/2026.
//

#ifndef GPT_2_FROM_SCRATCH_MULTIHEADATTENTION_H
#define GPT_2_FROM_SCRATCH_MULTIHEADATTENTION_H
#include "Attention.h"

class MultiHeadAttention {

    size_t out;
    std::vector<Attention> multi_attention;

    public:

    MultiHeadAttention(size_t n_heads, size_t batch_size, size_t sequence_len, size_t embedding_dim);

    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
};

#endif //GPT_2_FROM_SCRATCH_MULTIHEADATTENTION_H
