//
// Created by HP on 10/6/2026.
//

#ifndef GPT_2_FROM_SCRATCH_ATTENTION_H
#define GPT_2_FROM_SCRATCH_ATTENTION_H
#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Layers/include/Softmax.h"
#include "../../NumCPP/include/Tensor.h"

class Attention : public Layer {

    size_t batch_size;
    size_t sequence_len;
    size_t embedding_dim;

    Linear q_linear;
    Linear k_linear;
    Linear v_linear;
    Linear out_linear;

    Tensor att_weights;
    Tensor value;
    Tensor key;
    Tensor query;

    Softmax softmax;

    public:

    Attention(size_t batch_size, size_t sequence_len, size_t embedding_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zeroGrad() override;
};

#endif //GPT_2_FROM_SCRATCH_ATTENTION_H
