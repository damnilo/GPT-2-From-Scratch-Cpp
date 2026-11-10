#pragma once

#include "Structs.h"

bool forwardDevice(
    GemmDevicePath path,
    const DeviceWeights& weights,
    DeviceWorkspace& ws,
    const int* host_ids,
    size_t T,
    float* host_probs,
    AttentionState* saved = nullptr
);
