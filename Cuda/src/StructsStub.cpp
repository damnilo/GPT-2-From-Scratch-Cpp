
#include "../include/Structs.h"

bool uploadDeviceWeights(const Model&, DeviceWeights&) {
    return false;
}

void freeDeviceWeights(DeviceWeights&) {}

bool allocDeviceWorkspace(const ModelConfig&, DeviceWorkspace&) {
    return false;
}

void freeDeviceWorkspace(DeviceWorkspace&) {}

bool allocDeviceBackwardWs(size_t C, size_t n_heads, size_t capacity, DeviceBackwardWs& ws) {
    return false;
}

void freeDeviceBackwardWs(DeviceBackwardWs& ws) {}

void freeAttentionState(AttentionState&) {}

bool uploadAttentionWeights(MultiHeadAttention&, AttentionState&) {
    return false;
}

bool prepareAttentionActivations(AttentionState&, size_t, size_t, size_t, size_t) {
    return false;
}

bool prepareAttentionState(AttentionState&, size_t, size_t, size_t, size_t) {
    return false;
}
