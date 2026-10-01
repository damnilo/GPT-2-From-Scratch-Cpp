//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_EMBEDDING_H
#define GPT_2_FROM_SCRATCH_EMBEDDING_H

#include "../../NumCPP/include/Tensor.h"

class Embedding {
    Tensor weight;

    public:
    Embedding(size_t input_size, size_t output_size);
    Tensor forward(const std::vector<int>& tokens);
};

#endif //GPT_2_FROM_SCRATCH_EMBEDDING_H
