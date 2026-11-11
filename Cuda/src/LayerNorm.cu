
#include "../include/LayerNorm.h"
#include "../include/Kernels.cuh"

#include <cuda_runtime.h>

bool forwardLayerNormDevice(
    LayerNorm& layer,
    const float* host_in,
    size_t T,
    LayerNormState& state,
    std::vector<float>& host_y
) {
    if (host_in == nullptr || T == 0 || T > 1024) {
        return false;
    }

    if (!state.owns_weights) {
        if (!uploadLayerNormWeights(layer, state)) {
            return false;
        }
    }

    size_t C = state.C;
    if (C == 0 || C > 1024 || state.gamma == nullptr || state.beta == nullptr) {
        return false;
    }

    if (!prepareLayerNormActivation(state, T, C)) {
        return false;
    }

    if (cudaMemcpy(state.x, host_in, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    } 

    cudaLayerNormForward<<<T, C>>>(state.x, state.norm, state.var, state.y, state.gamma, state.beta, T, C);

    host_y.resize(T * C);
    if (cudaMemcpy(host_y.data(), state.y, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    return true;
}

bool backwardLayerNormDevice(
    LayerNormState& state,
    LayerNorm& layer,
    const float* host_grad,
    LayerNormGrad& out
) {
    size_t T = state.T;
    size_t C = state.C;

    if (host_grad == nullptr || T == 0 || T > 1024 || C == 0 || C > 1024) {
        return false;
    }

    if (state.x == nullptr
        || state.norm == nullptr
        || state.var == nullptr
        || state.y == nullptr
        || state.gamma == nullptr
        || state.beta == nullptr
        || state.dgamma == nullptr
        || state.dbeta == nullptr
    ) {
        return false;
    }

    if (layer.parameters().size() != 2) {
        return false;
    }

    if (cudaMemcpy(state.x, host_grad, T * C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    cudaLayerNormGrad<<<1, C>>>(state.x, state.norm, state.dgamma, state.dbeta, T, C);
    cudaLayerNormBackward<<<T, C>>>(state.x, state.norm, state.var, state.gamma, state.y, T, C);

    out.input.resize(T * C);
    out.gamma.resize(C);
    out.beta.resize(C);

    if (cudaMemcpy(out.input.data(), state.y, T * C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(out.gamma.data(), state.dgamma, C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(out.beta.data(), state.dbeta, C * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    return true;
}