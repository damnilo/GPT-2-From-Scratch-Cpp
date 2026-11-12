#pragma once

#include <vector>

#include "Structs.h"
#include "../../Transformer/include/Model.h"

class AdamW;

struct StackDevice {
    GemmDevicePath path = GemmDevicePath::Custom;
    std::vector<AttentionDevice> attention;
    std::vector<FeedForwardDevice> feed;
    std::vector<LayerNormDevice> norms;
    VocabDevice vocab;
    float* x = nullptr;
    float* y = nullptr;
    float* grad = nullptr;
    float* spare = nullptr;
    size_t T = 0;
    size_t capacity = 0;
    size_t C = 0;
};

bool enableDeviceStack(Model& model, GemmDevicePath path, StackDevice& stack);
void disableDeviceStack(Model& model, StackDevice& stack);
bool refreshDeviceStack(Model& model, StackDevice& stack);

bool forwardStackDevice(
    StackDevice& stack,
    const Model& model,
    const float* host_embed,
    size_t T,
    std::vector<float>& host_logits
);

bool backwardStackDevice(
    StackDevice& stack,
    Model& model,
    const float* host_logit_grad,
    std::vector<float>& host_embed_grad
);

bool adamDevice(StackDevice& stack, const AdamW& optimizer);
bool loadAdamState(const Model& model, StackDevice& stack, const AdamW& optimizer);
bool storeAdamState(Model& model, StackDevice& stack, AdamW& optimizer);
