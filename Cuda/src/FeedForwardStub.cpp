
#include "../include/FeedForward.h"


bool forwardFeedForwardDevice(
    GemmDevicePath path,
    FeedForward& layer,
    const float* host_in,
    size_t T,
    FeedForwardState& state,
    std::vector<float>& host_out
) {
    return false;
}

bool backwardFeedForwardDevice(
    GemmDevicePath path,
    const FeedForwardState& state,
    FeedForward& layer,
    const float* host_grad,
    FeedForwardGrad& out
) {
    return false;
}

bool refreshFeedForwardWeights(
    Model& model,
    std::vector<FeedForwardDevice>& slots
) {
    return false;
}

bool enableDeviceFeedForward(
    Model& model,
    GemmDevicePath path,
    std::vector<FeedForwardDevice>& slots
) {
    return false;
}

void disableDeviceFeedForward(
    Model& model,
    std::vector<FeedForwardDevice>& slots
) {}