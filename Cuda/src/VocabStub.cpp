
#include "../include/Vocab.h"


bool forwardVocabDevice(
    GemmDevicePath path,
    Linear& layer,
    const float* host_in,
    size_t T,
    VocabState& state,
    std::vector<float>& host_logits
) {
    return false;
}

bool backwardVocabDevice(
    GemmDevicePath path,
    const VocabState& state,
    Linear& layer,
    const float* host_grad,
    VocabGrad* out,
    float*
) {
    return false;
}

bool softmaxCELossDevice(VocabState&, const int*, float&) {
    return false;
}

bool forwardVocabOnDevice(
    GemmDevicePath,
    Linear&,
    const float*,
    size_t,
    VocabState&
) {
    return false;
}

bool refreshVocabWeights(Model& model, VocabDevice& slot) {
    return false;
}

bool enableDeviceVocab(
    Model& model,
    GemmDevicePath path,
    VocabDevice& slot
) {
    return false;
}

void disableDeviceVocab(Model& model, VocabDevice& slot) {}