#pragma once

#include <cstddef>

#include "../../NumCPP/include/DeviceGemm.h"

// Device W is [in, out]. y and grad are [rows, out]. x and dx are [rows, in].
// device_dW matches W. device_db and host_db are [out]. host_dW is [out, in].
// A batch packs that layout per matrix. Pass either pair, or both.
// The backward scratch stays on the device.

inline size_t linearBackwardScratch(size_t rows, size_t in, size_t out) {
    return in * out * 2 + in * rows + out;
}

inline size_t linearBatchBackwardScratch(size_t batches, size_t rows, size_t in, size_t out) {
    return batches * linearBackwardScratch(rows, in, out);
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
);

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
    float* device_dW = nullptr,
    float* device_db = nullptr
);

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
);

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
    float* device_dW = nullptr,
    float* device_db = nullptr
);
