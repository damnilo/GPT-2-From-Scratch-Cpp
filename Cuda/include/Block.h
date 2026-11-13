#pragma once

#include "Structs.h"
#include "../../Transformer/include/Model.h"

class AdamW;

bool enableDeviceStack(Model& model, GemmDevicePath path, StackDevice& stack);
void disableDeviceStack(Model& model, StackDevice& stack);
bool refreshDeviceStack(Model& model, StackDevice& stack);

bool forwardStackDevice(
    StackDevice& stack,
    const Model& model,
    const int* host_ids,
    size_t T
);

bool lossStackDevice(StackDevice& stack, const int* host_targets, float& loss);

bool backwardStackDevice(
    StackDevice& stack,
    Model& model,
    const float* host_logit_grad
);

bool adamDevice(StackDevice& stack, const AdamW& optimizer);
bool loadAdamState(const Model& model, StackDevice& stack, const AdamW& optimizer);
bool storeAdamState(Model& model, StackDevice& stack, AdamW& optimizer);
bool pullDeviceGradients(const Model& model, const StackDevice& stack);
