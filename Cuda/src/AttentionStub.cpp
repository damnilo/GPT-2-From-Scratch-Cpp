#include "../include/Attention.h"

bool forwardAttentionDevice(GemmDevicePath, MultiHeadAttention&, const float*, size_t, AttentionState&) {
    return false;
}

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

bool enableDeviceAttention(Model&, GemmDevicePath, std::vector<AttentionDevice>&) {
    return false;
}

void disableDeviceAttention(Model&, std::vector<AttentionDevice>&) {}
