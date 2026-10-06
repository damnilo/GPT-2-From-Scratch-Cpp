//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_SEQUENTIAL_H
#define GPT_2_FROM_SCRATCH_SEQUENTIAL_H
#include <memory>
#include <vector>

#include "Layer.h"

class Sequential {
    std::vector<std::unique_ptr<Layer>> layers;

    public:

    Sequential() = default;
    void addLayer(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
};

#endif //GPT_2_FROM_SCRATCH_SEQUENTIAL_H