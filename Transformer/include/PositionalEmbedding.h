//
// Created by HP on 10/10/2026.
//

#ifndef GPT_2_FROM_SCRATCH_POSITIONALEMBEDDING_H
#define GPT_2_FROM_SCRATCH_POSITIONALEMBEDDING_H
#include "../../NeuralNet/Layers/include/Embedding.h"
#include "../../NeuralNet/Layers/include/Layer.h"

class PositionalEmbedding : public Layer {

    Embedding tokens;
    Embedding positions;

public:

    PositionalEmbedding(size_t vocab_size, size_t block_size, size_t embedding_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor &grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    void zeroGrad() override;
};

#endif //GPT_2_FROM_SCRATCH_POSITIONALEMBEDDING_H
