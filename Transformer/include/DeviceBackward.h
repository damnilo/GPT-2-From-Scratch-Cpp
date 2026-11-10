
#ifndef GPT_2_FROM_SCRATCH_DEVICEBACKWARD_H
#define GPT_2_FROM_SCRATCH_DEVICEBACKWARD_H

#include "MultiHeadAttention.h"
#include "DeviceWeights.h"
#include "../../NumCPP/include/DeviceGemm.h"

struct DeviceBackwardWs {
    float* x = nullptr;
    float* q = nullptr;
    float* v = nullptr;
    float* k = nullptr;
    float* kt = nullptr;
    float* scores = nullptr;
    float* ctx = nullptr;
    float* joined = nullptr;
    float* grad = nullptr;
    float* d_joined = nullptr;
    float* d_ctx = nullptr;
    float* d_v = nullptr;
    float* d_p = nullptr;
    float* d_scores = nullptr;
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_x = nullptr;
    float* d_input = nullptr;
    size_t capacity = 0;
    size_t C = 0;
    size_t d = 0;
    size_t n_heads = 0;
};

struct HeadGrad {
    std::vector<float> Wq, bq;
    std::vector<float> Wk, bk;
    std::vector<float> Wv, bv;
    std::vector<float> Wo, bo;
};

struct AttentionGrad {
    std::vector<float> input;
    std::vector<HeadGrad> heads;
    std::vector<float> Wo, bo;
};

struct AttentionState {
    float* Wq = nullptr;
    float* bq = nullptr;
    float* Wk = nullptr;
    float* bk = nullptr;
    float* Wv = nullptr;
    float* bv = nullptr;
    float* Wo_h = nullptr;
    float* bo_h = nullptr;
    float* Wo = nullptr;
    float* bo = nullptr;

    float* x = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* scores = nullptr;
    float* ctx = nullptr;
    float* joined = nullptr;

    size_t T = 0;
    size_t C = 0;
    size_t d = 0;
    size_t n_heads = 0;
    bool owns_weights = false;
};

void freeAttentionState(AttentionState& state);

bool allocDeviceBackwardWs(
    size_t C, 
    size_t n_heads, 
    size_t capacity, 
    DeviceBackwardWs& ws
);

void freeDeviceBackwardWs(DeviceBackwardWs& ws);

bool backwardAttentionDevice(
    GemmDevicePath path,
    const AttentionState& state,
    MultiHeadAttention& layer,
    DeviceBackwardWs& ws,
    const float* host_grad,
    AttentionGrad& out
);

#endif