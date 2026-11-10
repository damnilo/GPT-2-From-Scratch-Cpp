#include "../include/DeviceBackward.h"

bool allocDeviceBackwardWs(size_t C, size_t n_heads, size_t capacity, DeviceBackwardWs& ws) {
    return false;
}

void freeDeviceBackwardWs(DeviceBackwardWs& ws) {}

bool backwardAttentionDevice(
    GemmDevicePath,
    const AttentionState&,
    MultiHeadAttention&,
    DeviceBackwardWs&,
    const float*,
    AttentionGrad&
) {
    return false;
}