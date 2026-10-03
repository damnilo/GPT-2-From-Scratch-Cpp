//
// Created by HP on 10/3/2026.
//

#include "../include/Embedding.h"

Embedding::Embedding(size_t input_size, size_t output_size) {
    std::vector<size_t> shape = {input_size, output_size};
    this->weight = Tensor(shape);
    this->weight.randomize();
}

Tensor Embedding::forward(const std::vector<int>& tokens) {
    auto embedding_dim = this->weight.getShape()[1];
    std::vector<size_t> shape = {tokens.size(), embedding_dim};

    Tensor output(shape);

    for (size_t i = 0; i < tokens.size(); i++) {
        int token = tokens[i];

        for (size_t j = 0; j < embedding_dim; j++) {
            size_t weight_i = token * embedding_dim + j;
            size_t output_i = i * embedding_dim + j;

            output[output_i] = this->weight[weight_i];
        }
    }

    return output;
}