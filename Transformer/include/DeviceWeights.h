
#ifndef GPT_2_FROM_SCRATCH_DEVICEWEIGHTS_H
#define GPT_2_FROM_SCRATCH_DEVICEWEIGHTS_H

#include <cstddef>
#include <vector>

#include "Model.h"

class Model;

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
};

struct DeviceWeights {
    GpuTensor token_embed;
    GpuTensor pos_embed;
    std::vector<LayerWeights> layers;
    GpuTensor ln_f_gamma, ln_f_beta;
    GpuTensor W_vocab, b_vocab;
};

bool uploadDeviceWeights(const Model& model, DeviceWeights& out);

void freeDeviceWeights(DeviceWeights& weights);

#endif