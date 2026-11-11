#pragma once

#include <vector>

#include "Structs.h"
#include "../../Transformer/include/MultiHeadAttention.h"

bool forwardAttentionDevice(
    GemmDevicePath path,
    MultiHeadAttention& layer,
    const float* host_in,
    size_t T,
    AttentionState& state,
    std::vector<float>* host_out = nullptr
);

bool refreshAttentionWeights(
    Model& model,
    std::vector<AttentionDevice>& slots
);

bool backwardAttentionDevice(
    GemmDevicePath path,
    const AttentionState& state,
    MultiHeadAttention& layer,
    DeviceBackwardWs& ws,
    const float* host_grad,
    AttentionGrad& out
);

bool enableDeviceAttention(
    Model& model,
    GemmDevicePath path,
    std::vector<AttentionDevice>& slots
);

void disableDeviceAttention(
    Model& model,
    std::vector<AttentionDevice>& slots
);

bool forwardAttentionOnDevice(
    GemmDevicePath path,
    MultiHeadAttention& layer,
    const float* device_in,
    float* device_out,
    size_t T,
    AttentionState& state
);

bool backwardAttentionOnDevice(
    GemmDevicePath path,
    const AttentionState& state,
    MultiHeadAttention& layer,
    DeviceBackwardWs& ws,
    const float* device_grad,
    float* device_dx,
    AttentionGrad& out
);

void commitAttentionGrad(MultiHeadAttention& layer, const AttentionGrad& grad);
