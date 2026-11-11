
#include <cmath>
#include <cuda_runtime.h>

#include "Structs.h"

__global__ void cudaGather3d(
    float* scratch,
    float* heads,
    size_t H,
    size_t T,
    size_t C,
    size_t d
);

__global__ void cudaScatter(
    float* head,
    float* x,
    size_t h,
    size_t T,
    size_t C,
    size_t d
);

__global__ void cudaScatter3d(
    float* head,
    float* x,
    size_t H,
    size_t T,
    size_t C,
    size_t d
);

__global__ void cudaAdd(
    float* residual,
    float* scratch,
    float* x,
    size_t T,
    size_t C
);

__global__ void cudaAddVectors(
    float* a,
    const float* b,
    size_t n
);

__global__ void cudaAddBias(
    float* a,
    GpuTensor bias,
    size_t T,
    size_t d
);

__global__ void cudaAddBias3d(
    float* a,
    float* bias,
    size_t H,
    size_t T,
    size_t d
);

__global__ void cudaSumRows(
    float* a,
    float* bias,
    size_t H,
    size_t T, 
    size_t d
);

__global__ void cudaSumRowsV(
    const float* grad,
    float* db,
    size_t T,
    size_t V
);

__global__ void cudaTranspose(
    float* a,
    float* a_t,
    size_t T,
    size_t d
);

__global__ void cudaTransposeCV(
    const float* in,
    float* out,
    size_t C,
    size_t V
);


__global__ void cudaTranspose3d(
    float* a,
    float* a_t,
    size_t H,
    size_t T,
    size_t d
);

__global__ void cudaEmbed(
    int* ids, 
    GpuTensor token_embed, 
    GpuTensor pos_embed, 
    float* x, 
    size_t T, 
    size_t C
);

__global__ void cudaLayernorm(
    float* x, 
    float* y, 
    GpuTensor gamma, 
    GpuTensor beta, 
    size_t T, 
    size_t C
);

__global__ void cudaMaskSoftmax(
    float* scores,
    size_t T,
    float scale
);

__global__ void cudaMaskSoftmax3d(
    float* scores,
    size_t H,
    size_t T,
    float scale
);

__global__ void cudaSoftmaxBackward3d(
    float* scores,
    float* d_p,
    float* d_scores,
    size_t H,
    size_t T,
    float scale
);

__global__ void cudaLayerNormForward(
    const float* x,
    float* norm,
    float* var,
    float* y,
    const float* gamma,
    const float* beta,
    size_t T,
    size_t C
);

__global__ void cudaLayerNormGrad(
    const float* grad,
    const float* norm,
    float* dgamma,
    float* dbeta,
    size_t T,
    size_t C
);

__global__ void cudaLayerNormBackward(
    const float* grad,
    const float* norm,
    const float* var,
    const float* gamma,
    float* dx,
    size_t T,
    size_t C
);

__global__ void cudaGELU(
    float* x,
    size_t T,
    size_t F
);

__global__ void cudaGELU(
    const float* x,
    float* y,
    size_t T,
    size_t F
);

__global__ void cudaGELUBackward(
    const float* hidden,
    float* grad,
    size_t n
);