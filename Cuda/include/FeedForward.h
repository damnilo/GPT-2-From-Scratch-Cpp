#pragma once

#include <vector>

#include "Structs.h"
#include "../../Transformer/include/FeedForward.h"


bool forwardFeedForwardDevice(
    GemmDevicePath path,
    FeedForward& layer,
    const float* host_in,
    size_t T,
    FeedForwardState& state,
    std::vector<float>& host_out
);

bool backwardFeedForwardDevice(
    GemmDevicePath path,
    const FeedForwardState& state,
    FeedForward& layer,
    const float* host_grad,
    FeedForwardGrad& out
);

bool refreshFeedForwardWeights(
    Model& model,
    std::vector<FeedForwardDevice>& slots
);

bool enableDeviceFeedForward(
    Model& model,
    GemmDevicePath path,
    std::vector<FeedForwardDevice>& slots
);

void disableDeviceFeedForward(
    Model& model,
    std::vector<FeedForwardDevice>& slots
);

bool forwardFeedForwardOnDevice(
    GemmDevicePath path,
    FeedForward& layer,
    const float* device_in,
    float* device_out,
    size_t T,
    FeedForwardState& state
);

bool backwardFeedForwardOnDevice(
    GemmDevicePath path,
    const FeedForwardState& state,
    FeedForward& layer,
    const float* device_grad,
    float* device_dx,
    FeedForwardGrad& out
);

void commitFeedForwardGrad(FeedForward& layer, const FeedForwardGrad& grad);
