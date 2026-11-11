#pragma once

#include <vector>

#include "Structs.h"
#include "../../NeuralNet/Layers/include/Linear.h"

bool forwardVocabDevice(
    GemmDevicePath path,
    Linear& layer,
    const float* host_in,
    size_t T,
    VocabState& state,
    std::vector<float>& host_logits
);

bool backwardVocabDevice(
    GemmDevicePath path,
    const VocabState& state,
    Linear& layer,
    const float* host_grad,
    VocabGrad& out,
    float* device_dx = nullptr
);

bool forwardVocabOnDevice(
    GemmDevicePath path,
    Linear& layer,
    const float* device_in,
    size_t T,
    VocabState& state,
    std::vector<float>& host_logits
);

bool refreshVocabWeights(Model& model, VocabDevice& slot);

bool enableDeviceVocab(
    Model& model,
    GemmDevicePath path,
    VocabDevice& slot
);

void disableDeviceVocab(Model& model, VocabDevice& slot);
