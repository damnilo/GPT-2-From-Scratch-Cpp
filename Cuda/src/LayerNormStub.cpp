
#include "../include/LayerNorm.h"


bool forwardLayerNormDevice(
    LayerNorm& layer,
    const float* host_in,
    size_t T,
    LayerNormState& state,
    std::vector<float>& host_y
) {
    return false;
}

bool backwardLayerNormDevice(
    LayerNormState& state,
    LayerNorm& layer,
    const float* host_grad,
    LayerNormGrad& out
) {
    return false;
}