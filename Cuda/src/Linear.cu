#include "../include/Linear.h"
#include "../include/Kernels.cuh"

namespace {

unsigned blocks(size_t n) {
    return static_cast<unsigned>((n + 255) / 256);
}

}

bool linearForwardDevice(
    GemmDevicePath path,
    const float* x,
    const float* W,
    const float* bias,
    float* y,
    size_t rows,
    size_t in,
    size_t out
) {
    if (x == nullptr || W == nullptr || bias == nullptr || y == nullptr || rows == 0 || in == 0 || out == 0) {
        return false;
    }
    if (!gemmDevice(path, x, W, y, rows, in, out)) {
        return false;
    }
    const GpuTensor b{const_cast<float*>(bias), out};
    cudaAddBias<<<dim3(blocks(out), static_cast<unsigned>(rows)), 256>>>(y, b, rows, out);
    return true;
}

bool linearBackwardDevice(
    GemmDevicePath path,
    const float* x,
    const float* W,
    const float* grad,
    float* dx,
    float* host_dW,
    float* host_db,
    float* scratch,
    size_t rows,
    size_t in,
    size_t out,
    float* device_dW,
    float* device_db
) {
    const bool to_host = host_dW != nullptr || host_db != nullptr;
    const bool to_device = device_dW != nullptr || device_db != nullptr;
    if (x == nullptr || W == nullptr || grad == nullptr || dx == nullptr || scratch == nullptr
        || rows == 0 || in == 0 || out == 0
        || (host_dW == nullptr) != (host_db == nullptr)
        || (device_dW == nullptr) != (device_db == nullptr)
        || (!to_host && !to_device)) {
        return false;
    }

    float* wt = scratch;
    float* dw = wt + in * out;
    float* xt = dw + in * out;
    float* db = xt + in * rows;

    cudaTransposeCV<<<dim3(blocks(out), static_cast<unsigned>(in)), 256>>>(W, wt, in, out);
    if (!gemmDevice(path, grad, wt, dx, rows, out, in)) {
        return false;
    }

    cudaTransposeCV<<<dim3(blocks(in), static_cast<unsigned>(rows)), 256>>>(x, xt, rows, in);
    if (!gemmDevice(path, xt, grad, dw, in, rows, out)) {
        return false;
    }
    if (device_dW != nullptr
        && cudaMemcpy(device_dW, dw, in * out * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }

    cudaSumRowsV<<<blocks(out), 256>>>(grad, db, rows, out);
    if (device_db != nullptr
        && cudaMemcpy(device_db, db, out * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }
    if (host_dW == nullptr) {
        return true;
    }

    cudaTransposeCV<<<dim3(blocks(out), static_cast<unsigned>(in)), 256>>>(dw, wt, in, out);
    if (cudaMemcpy(host_dW, wt, in * out * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(host_db, db, out * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool linearBatchForwardDevice(
    GemmDevicePath path,
    const float* x,
    const float* W,
    const float* bias,
    float* y,
    size_t batches,
    size_t rows,
    size_t in,
    size_t out
) {
    if (x == nullptr || W == nullptr || bias == nullptr || y == nullptr
        || batches == 0 || rows == 0 || in == 0 || out == 0) {
        return false;
    }
    if (!gemmBatchDevice(path, x, W, y, batches, rows, in, out)) {
        return false;
    }
    cudaAddBias3d<<<dim3(blocks(out), static_cast<unsigned>(batches * rows)), 256>>>(
        y, const_cast<float*>(bias), batches, rows, out);
    return true;
}

bool linearBatchBackwardDevice(
    GemmDevicePath path,
    const float* x,
    const float* W,
    const float* grad,
    float* dx,
    float* host_dW,
    float* host_db,
    float* scratch,
    size_t batches,
    size_t rows,
    size_t in,
    size_t out,
    float* device_dW,
    float* device_db
) {
    const bool to_host = host_dW != nullptr || host_db != nullptr;
    const bool to_device = device_dW != nullptr || device_db != nullptr;
    if (x == nullptr || W == nullptr || grad == nullptr || dx == nullptr || scratch == nullptr
        || batches == 0 || rows == 0 || in == 0 || out == 0
        || (host_dW == nullptr) != (host_db == nullptr)
        || (device_dW == nullptr) != (device_db == nullptr)
        || (!to_host && !to_device)) {
        return false;
    }

    const size_t weight_n = in * out;
    const size_t x_n = rows * in;
    const size_t grad_n = rows * out;
    float* wt = scratch;
    float* dw = wt + batches * weight_n;
    float* xt = dw + batches * weight_n;
    float* db = xt + batches * in * rows;

    for (size_t batch = 0; batch < batches; ++batch) {
        cudaTransposeCV<<<dim3(blocks(out), static_cast<unsigned>(in)), 256>>>(
            W + batch * weight_n, wt + batch * weight_n, in, out);
    }
    if (!gemmBatchDevice(path, grad, wt, dx, batches, rows, out, in)) {
        return false;
    }

    for (size_t batch = 0; batch < batches; ++batch) {
        cudaTransposeCV<<<dim3(blocks(in), static_cast<unsigned>(rows)), 256>>>(
            x + batch * x_n, xt + batch * in * rows, rows, in);
    }
    if (!gemmBatchDevice(path, xt, grad, dw, batches, in, rows, out)) {
        return false;
    }
    if (device_dW != nullptr
        && cudaMemcpy(device_dW, dw, batches * weight_n * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }

    for (size_t batch = 0; batch < batches; ++batch) {
        cudaSumRowsV<<<blocks(out), 256>>>(grad + batch * grad_n, db + batch * out, rows, out);
    }
    if (device_db != nullptr
        && cudaMemcpy(device_db, db, batches * out * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return false;
    }
    if (host_dW == nullptr) {
        return true;
    }

    for (size_t batch = 0; batch < batches; ++batch) {
        cudaTransposeCV<<<dim3(blocks(out), static_cast<unsigned>(in)), 256>>>(
            dw + batch * weight_n, wt + batch * weight_n, in, out);
    }
    if (cudaMemcpy(host_dW, wt, batches * weight_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(host_db, db, batches * out * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}
