#pragma once

#include <vector>

#include "Structs.h"
#include "../../NeuralNet/Layers/include/LayerNorm.h"

bool forwardLayerNormDevice(
    LayerNorm& layer,
    const float* host_in,
    size_t T,
    LayerNormState& state,
    std::vector<float>& host_y
);

bool backwardLayerNormDevice(
    LayerNormState& state,
    LayerNorm& layer,
    const float* host_grad,
    LayerNormGrad& out
);

bool forwardLayerNormOnDevice(
    LayerNorm& layer,
    const float* device_in,
    float* device_out,
    size_t T,
    LayerNormState& state
);

bool backwardLayerNormOnDevice(
    LayerNormState& state,
    LayerNorm& layer,
    const float* device_grad,
    float* device_dx,
    LayerNormGrad& out
);

void commitLayerNormGrad(LayerNorm& layer, const LayerNormGrad& grad);
