#include "../include/Block.h"

bool enableDeviceStack(Model&, GemmDevicePath, StackDevice&) {
    return false;
}

void disableDeviceStack(Model& model, StackDevice&) {
    model.useStack(nullptr);
}

bool refreshDeviceStack(Model&, StackDevice&) {
    return false;
}

bool forwardStackDevice(StackDevice&, const Model&, const float*, size_t, std::vector<float>&) {
    return false;
}

bool backwardStackDevice(StackDevice&, Model&, const float*, std::vector<float>&) {
    return false;
}
