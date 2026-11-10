
#pragma once

#include <vector>

#include "../../NumCPP/include/DeviceGemm.h"
#include "../../Transformer/include/Model.h"

class Model;

class MultiHeadAttention;

struct GpuTensor {
    float* ptr = nullptr;
    size_t count = 0;
};

struct HeadWeights {
    GpuTensor Wq, bq;
    GpuTensor Wk, bk;
    GpuTensor Wv, bv;
    GpuTensor Wo, bo;
};

struct LayerWeights {
    GpuTensor ln1_gamma, ln1_beta;
    std::vector<HeadWeights> heads;
    GpuTensor Wo, bo;
    GpuTensor ln2_gamma, ln2_beta;
    GpuTensor W1, b1;
    GpuTensor W2, b2;
    GpuTensor Wq_all, bq_all;
    GpuTensor Wk_all, bk_all;
    GpuTensor Wv_all, bv_all;
    GpuTensor Wo_all, bo_all;
};

struct DeviceWeights {
    GpuTensor token_embed;
    GpuTensor pos_embed;
    std::vector<LayerWeights> layers;
    GpuTensor ln_f_gamma, ln_f_beta;
    GpuTensor W_vocab, b_vocab;
};

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
    size_t n_heads = 0;
};

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

struct AttentionDevice {
    GemmDevicePath path = GemmDevicePath::Custom;
    AttentionState state;
    DeviceBackwardWs ws;
};

bool uploadDeviceWeights(const Model& model, DeviceWeights& out);

void freeDeviceWeights(DeviceWeights& weights);

bool allocDeviceWorkspace(const ModelConfig& config, DeviceWorkspace& ws);

void freeDeviceWorkspace(DeviceWorkspace& ws);

bool allocDeviceBackwardWs(
    size_t C, 
    size_t n_heads, 
    size_t capacity, 
    DeviceBackwardWs& ws
);

void freeDeviceBackwardWs(DeviceBackwardWs& ws);

void freeAttentionState(AttentionState& state);

bool prepareAttentionState(AttentionState& state, size_t H, size_t T, size_t C, size_t d);

bool copyDevice(float* dest, const float* src, size_t count);
