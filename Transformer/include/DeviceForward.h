
#ifndef GPT_2_FROM_SCRATCH_DEVICEFORWARD_H
#define GPT_2_FROM_SCRATCH_DEVICEFORWARD_H

#include "Model.h"
#include "DeviceWeights.h"
#include "../../NumCPP/include/DeviceGemm.h"

class Model;

struct DeviceWorkspace {
    int* ids = nullptr;
    float* x = nullptr;
    float* residual = nullptr;
    float* scratch = nullptr;
    float* head = nullptr;
    float* q = nullptr; 
    float* k = nullptr;
    float* v = nullptr;
    float* kt = nullptr;
    float* scores = nullptr;
    float* ctx = nullptr;
    float* hidden = nullptr;
    float* row = nullptr;
    float* logits = nullptr;
    size_t capacity = 0;
    size_t C = 0;
    size_t V = 0;
    size_t d = 0;
    size_t F = 0;
};

bool allocDeviceWorkspace(const ModelConfig& config, DeviceWorkspace& ws);

void freeDeviceWorkspace(DeviceWorkspace& ws);

bool forwardDevice(
    GemmDevicePath path, 
    const DeviceWeights& weights, 
    DeviceWorkspace& ws, 
    const int* host_ids, 
    size_t T, 
    float* host_probs
);

#endif