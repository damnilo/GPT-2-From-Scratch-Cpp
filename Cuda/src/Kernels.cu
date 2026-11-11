
#include "../include/Kernels.cuh"


__global__ void cudaGather(
    float* scratch,
    float* head,
    size_t h,
    size_t T,
    size_t C,
    size_t d
) {
    int t = blockIdx.x;
    int k = threadIdx.x;

    if (t >= T || k >= d) return;

    head[t * d + k] = scratch[t * C + h * d + k];
}

__global__ void cudaGather3d(
    float* scratch,
    float* heads,
    size_t H,
    size_t T,
    size_t C,
    size_t d
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int k = threadIdx.x;
    
    if (t >= T || h >= H || k >= d) return;

    heads[h * T * d + t * d + k] = scratch[t * C + h * d + k];
}

__global__ void cudaScatter(
    float* head,
    float* x,
    size_t h,
    size_t T,
    size_t C,
    size_t d
) {
    int t = blockIdx.x;
    int k = threadIdx.x;

    if (t >= T || k >= d) return;

    x[t * C + h * d + k] = head[t * d + k];
}

__global__ void cudaScatter3d(
    float* head,
    float* x,
    size_t H,
    size_t T,
    size_t C,
    size_t d
) {

    int t = blockIdx.x;
    int h = blockIdx.y;
    int k = threadIdx.x;

    if (t >= T || h >= H || k >= d) return;

    x[t * C + h * d + k] = head[h * T * d + t * d + k];
}

__global__ void cudaAdd(
    float* residual,
    float* scratch,
    float* x,
    size_t T,
    size_t C
) {
    int t = blockIdx.x;
    int c = threadIdx.x;

    if (t >= T || c >= C) return;

    x[t * C + c] = residual[t * C + c] + scratch[t * C + c];
}

__global__ void cudaAddBias(
    float* a,
    GpuTensor bias,
    size_t T,
    size_t d
) {
    const size_t t = blockIdx.y;
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (t >= T || k >= d) return;

    a[t * d + k] += bias.ptr[k];
}

__global__ void cudaAddBias3d(
    float* a,
    float* bias,
    size_t H,
    size_t T,
    size_t d
) {

    int h = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= H * T || k >= d) return;

    a[h * d + k] += bias[(h / T) * d + k];
}

__global__ void cudaAddVectors(
    float* a,
    const float* b,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void cudaSumRows(
    float* a,
    float* bias,
    size_t H,
    size_t T, 
    size_t d
) {
    int h = blockIdx.x;
    int k = threadIdx.x;

    if (h >= H || k >= d) return;

    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        sum += a[h * d * T + i * d + k];
    }

    bias[h * d + k] = sum;
}

__global__ void cudaSumRowsV(
    const float* grad,
    float* db,
    size_t T,
    size_t V
) {
    const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= V) {
        return;
    }
    float sum = 0.0f;
    for (size_t t = 0; t < T; ++t) {
        sum += grad[t * V + v];
    }
    db[v] = sum;
}

__global__ void cudaTranspose(
    float* a,
    float* a_t,
    size_t T,
    size_t d
) {
    int t = blockIdx.x;
    int k = threadIdx.x;

    if (t >= T || k >= d) return;

    a_t[k * T + t] = a[t * d + k];
}

__global__ void cudaTransposeCV(
    const float* in,
    float* out,
    size_t C,
    size_t V
) {
    const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t c = blockIdx.y;
    if (v >= V || c >= C) {
        return;
    }
    out[v * C + c] = in[c * V + v];
}

__global__ void cudaTranspose3d(
    float* a,
    float* a_t,
    size_t H,
    size_t T,
    size_t d
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int k = threadIdx.x;

    if (t >= T || h >= H || k >= d) return;

    a_t[h * T * d + k * T + t] = a[h * T * d + t * d + k];
}

__global__ void cudaEmbed(
    int* ids, 
    GpuTensor token_embed, 
    GpuTensor pos_embed, 
    float* x, 
    size_t T, 
    size_t C
) {
    int t = blockIdx.x;
    int c = threadIdx.x;

    if (t >= T || c >= C) return;

    int id = ids[t];
    x[t * C + c] = token_embed.ptr[id * C + c] + pos_embed.ptr[t * C + c];
}

__global__ void cudaLayernorm(
    float* x, 
    float* y, 
    GpuTensor gamma, 
    GpuTensor beta, 
    size_t T, 
    size_t C
) {
    int t = blockIdx.x;
    int c = threadIdx.x;

    if (t >= T || c >= C) return;

    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += x[t * C + i];
    }

    float mean = (float)sum / C;

    sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += powf(x[t * C + i] - mean, 2);
    }

    float var = (float)sum / C;
    float inv = rsqrtf(var + 1e-5f);
    
    y[t * C + c] = gamma.ptr[c] * (x[t * C + c] - mean) * inv + beta.ptr[c];
}

__global__ void cudaMaskSoftmax(
    float* scores,
    size_t T,
    float scale
) {
    int t = blockIdx.x;

    if (t >= T) return;

    float* row = scores + t * T;

    for (size_t i = 0; i < T; ++i) {
        row[i] = (i <= t) ? row[i] * scale : -INFINITY;
    }

    float max = -INFINITY;
    for (size_t i = 0; i < T; ++i) {
        if (max < row[i]) {
            max = row[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < T; ++i) {
        sum += (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max);
    }

    for (size_t i = 0; i < T; ++i) {
        row[i] = (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max) / sum;
    }
}

__global__ void cudaMaskSoftmax3d(
    float* scores,
    size_t H,
    size_t T,
    float scale
) {

    int t = blockIdx.x;
    int h = blockIdx.y;

    if (t >= T || h >= H) return;

    float* row = scores + (h * T + t) * T;

    for (int i = 0; i < T; ++i) {
        row[i] = (i <= t) ? row[i] * scale : -INFINITY;
    } 

    float max = -INFINITY;
    for (int i = 0; i < T; ++i) {
        if (max < row[i]) max = row[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        sum += (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max);
    }

    for (int i = 0; i < T; ++i) {
        row[i] = (row[i] == -INFINITY) ? 0.0f : expf(row[i] - max) / sum;
    }
}

__global__ void cudaSoftmaxBackward3d(
    float* scores,
    float* d_p,
    float* d_scores,
    size_t H,
    size_t T,
    float scale
) {
    int t = blockIdx.x;
    int h = blockIdx.y;

    if (t >= T || h >= H) return;

    int row = (h * T + t) * T;
    float* tmp = d_scores + row;

    for (int i = 0; i < T; ++i) {
        tmp[i] = d_p[row + i] * scores[row + i];
    }

    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
        sum += tmp[i];
    }

    for (int i = 0; i < T; ++i) {
        d_scores[row + i] = scores[row + i] * (d_p[row + i] - sum) * scale;
    }
}

__global__ void cudaLayerNormForward(
    const float* x,
    float* norm,
    float* var,
    float* y,
    const float* gamma,
    const float* beta,
    size_t T,
    size_t C
) {
    const int t = blockIdx.x;
    const int c = threadIdx.x;

    __shared__ float mean;
    __shared__ float inv_std;

    if (t >= static_cast<int>(T)) return;

    if (c == 0) {
        float sum = 0.0f;
        for (int j = 0; j < static_cast<int>(C); ++j) {
            sum += x[t * C + j];
        }
        mean = sum / static_cast<float>(C);

        sum = 0.0f;
        for (int j = 0; j < static_cast<int>(C); ++j) {
            const float diff = x[t * C + j] - mean;
            sum += diff * diff;
        }
        const float v = sum / static_cast<float>(C);
        var[t] = v;
        inv_std = 1.0f / sqrtf(v + 1e-5f);
    }

    __syncthreads();

    if (c >= static_cast<int>(C)) return;

    const float n = (x[t * C + c] - mean) * inv_std;
    norm[t * C + c] = n;
    y[t * C + c] = gamma[c] * n + beta[c];
}

__global__ void cudaLayerNormGrad(
    const float* grad,
    const float* norm,
    float* dgamma,
    float* dbeta,
    size_t T,
    size_t C
) {
    const int c = threadIdx.x;

    if (c >= static_cast<int>(C)) return;

    float sum_b = 0.0f;
    float sum_g = 0.0f;

    for (int t = 0; t < static_cast<int>(T); ++t) {
        const float g = grad[static_cast<size_t>(t) * C + c];
        sum_b += g;
        sum_g += g * norm[static_cast<size_t>(t) * C + c];
    }

    dbeta[c] = sum_b;
    dgamma[c] = sum_g;
}

__global__ void cudaLayerNormBackward(
    const float* grad,
    const float* norm,
    const float* var,
    const float* gamma,
    float* dx,
    size_t T,
    size_t C
) {
    const int t = blockIdx.x;
    const int c = threadIdx.x;

    __shared__ float sum_dxhat;
    __shared__ float sum_dxhat_norm;
    __shared__ float inv_std;

    if (t >= static_cast<int>(T)) return;

    if (c == 0) {
        sum_dxhat = 0.0f;
        sum_dxhat_norm = 0.0f;

        for (int j = 0; j < static_cast<int>(C); ++j) {
            const float dxhat = grad[static_cast<size_t>(t) * C + j] * gamma[j];
            sum_dxhat += dxhat;
            sum_dxhat_norm += dxhat * norm[static_cast<size_t>(t) * C + j];
        }

        inv_std = 1.0f / sqrtf(var[t] + 1e-5f);
    }

    __syncthreads();

    if (c >= static_cast<int>(C)) return;

    const float dxhat = grad[static_cast<size_t>(t) * C + c] * gamma[c];
    const float inv_n = 1.0f / static_cast<float>(C);
    dx[static_cast<size_t>(t) * C + c] = inv_std * (dxhat - sum_dxhat * inv_n - norm[static_cast<size_t>(t) * C + c] * sum_dxhat_norm * inv_n);
}

__global__ void cudaGELU(
    float* x,
    size_t T,
    size_t F
) {
    int t = blockIdx.x;
    int f = threadIdx.x;

    if (t >= T || f >= F) return;

    float val = x[t * F + f];
    x[t * F + f] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + val * val * val * 0.044715f)));
}

__global__ void cudaGELU(
    const float* x,
    float* y,
    size_t T,
    size_t F
) {
    int t = blockIdx.x;
    int f = threadIdx.x;

    if (t >= T || f >= F) return;

    float val = x[t * F + f];
    y[t * F + f] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + val * val * val * 0.044715f)));
}

__device__ float geluTanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void cudaGELUBackward(const float* hidden, float* grad, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float h = sqrtf(1.19209290e-7f);
    const float x = hidden[i];
    grad[i] *= (geluTanh(x + h) - geluTanh(x - h)) / (2.0f * h);
}