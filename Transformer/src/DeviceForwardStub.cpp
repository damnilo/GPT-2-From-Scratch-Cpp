#include "../include/DeviceForward.h"

bool allocDeviceWorkspace(const ModelConfig&, DeviceWorkspace&) {
    return false;
}

void freeDeviceWorkspace(DeviceWorkspace&) {}

bool forwardDevice(GemmDevicePath, const DeviceWeights&, DeviceWorkspace&, const int*, size_t, float*) {
    return false;
}
