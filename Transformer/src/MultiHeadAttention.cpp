//
// Created by HP on 10/9/2026.
//
#include "../include/MultiHeadAttention.h"

MultiHeadAttention::MultiHeadAttention(size_t n_heads, size_t batch_size, size_t sequence_len, size_t embedding_dim) {
    this->out = static_cast<size_t>(embedding_dim / n_heads);

    for (size_t i = 0; i < n_heads; ++i) {
        Attention att(batch_size, sequence_len, out);
        multi_attention.push_back(att);
    }
}

Tensor MultiHeadAttention::forward(const Tensor &input) {
    Tensor ret = multi_attention[0].forward(input);

    for (size_t i = 1; i < multi_attention.size(); ++i) {
        ret = ret.concat(multi_attention[i].forward(input), -1);
    }

    return ret;
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output) {
    Tensor ret;

    for (size_t i = 0; i < multi_attention.size(); ++i) {
        Tensor head = grad_output.slice(-1, i * out, (i+1) * out);

        Tensor input_grad = multi_attention[i].backward(head);

        if (i == 0) {
            ret = input_grad;
        }else {
            ret = ret + input_grad;
        }
    }

    return ret;
}

