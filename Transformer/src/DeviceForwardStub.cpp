#include "../include/DeviceForward.h"
#include "../include/DeviceBackward.h"

bool allocDeviceWorkspace(const ModelConfig&, DeviceWorkspace&) {
    return false;
}

void freeDeviceWorkspace(DeviceWorkspace&) {}

void freeAttentionState(AttentionState&) {}

bool forwardDevice(GemmDevicePath, const DeviceWeights&, DeviceWorkspace&, const int*, size_t, float*, AttentionState*) {
    return false;
}

bool forwardAttentionDevice(GemmDevicePath, MultiHeadAttention&, const float*, size_t, AttentionState&) {
    return false;
}
