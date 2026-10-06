//
// Created by HP on 10/6/2026.
//
#include "../include/Attention.h"

#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Layers/include/Softmax.h"

Attention::Attention(size_t batch_size, size_t sequence_len, size_t embedding_dim) :
batch_size(batch_size),
sequence_len(sequence_len),
embedding_dim(embedding_dim),
q_linear(embedding_dim, embedding_dim),
v_linear(embedding_dim, embedding_dim),
k_linear(embedding_dim, embedding_dim),
out_linear(embedding_dim, embedding_dim),
softmax(-1)
{}

Tensor Attention::forward(const Tensor &input) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("input.size() != 3");
    }

    if (input.getShape()[0] != batch_size
        || input.getShape()[1] != sequence_len
        || input.getShape()[2] != embedding_dim) {
        throw std::invalid_argument("Shape missmatch");
    }

    auto d_k = static_cast<float>(embedding_dim);

    query = input;
    key = input;
    value = input;

    query = q_linear.forward(query);
    key = k_linear.forward(key);
    value = v_linear.forward(value);

    Tensor multiplied = query.batchMatmul(key.transpose(1, 2));
    multiplied = multiplied / (std::sqrt(d_k));

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < sequence_len; j++) {
            for (size_t k = j+1; k < sequence_len; k++) {
                multiplied.at({i, j, k}) = -std::numeric_limits<float>::infinity();
            }
        }
    }

    att_weights = softmax.forward(multiplied);
    Tensor output = att_weights.batchMatmul(value);
    output = out_linear.forward(output);

    return output;
}

Tensor Attention::backward(const Tensor &grad_output) {
    if (grad_output.ndim() != 3) {
        throw std::invalid_argument("grad_output.size() != 3");
    }

    if (grad_output.getShape()[0] !=batch_size
        || grad_output.getShape()[1] != sequence_len
        || grad_output.getShape()[2] != embedding_dim) {
        throw std::invalid_argument("Shape missmatch");
    }

    auto d_k = static_cast<float>(embedding_dim);

    Tensor grad_att = out_linear.backward(grad_output);

    Tensor grad_value = att_weights.transpose(1, 2).batchMatmul(grad_att);
    Tensor grad_att_weights = grad_att.batchMatmul(value.transpose(1, 2));

    Tensor grad_scores = softmax.backward(grad_att_weights);
    grad_scores = grad_scores / (std::sqrt(d_k));

    Tensor grad_query = grad_scores.batchMatmul(key);
    Tensor grad_key = grad_scores.transpose(1, 2).batchMatmul(query);

    grad_query = q_linear.backward(grad_query);
    grad_key = k_linear.backward(grad_key);
    grad_value = v_linear.backward(grad_value);

    return grad_value + grad_key + grad_query;
}

std::vector<Tensor *> Attention::parameters() {
    std::vector<std::vector<Tensor *>> list = {q_linear.parameters(), k_linear.parameters(), v_linear.parameters(), out_linear.parameters()};
    std::vector<Tensor*> parameters;
    for (const std::vector<Tensor*>& i : list) {
        for (Tensor* j : i) {
            parameters.push_back(j);
        }
    }

    return parameters;
}

std::vector<Tensor *> Attention::gradients() {
    std::vector<std::vector<Tensor *>> list = {q_linear.gradients(), k_linear.gradients(), v_linear.gradients(), out_linear.gradients()};
    std::vector<Tensor*> gradients;
    for (const std::vector<Tensor*>& i : list) {
        for (Tensor* j : i) {
            gradients.push_back(j);
        }
    }

    return gradients;
}
