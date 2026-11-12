#include "../include/Linear.h"

bool linearForwardDevice(
    GemmDevicePath,
    const float*,
    const float*,
    const float*,
    float*,
    size_t,
    size_t,
    size_t
) {
    return false;
}

bool linearBackwardDevice(
    GemmDevicePath,
    const float*,
    const float*,
    const float*,
    float*,
    float*,
    float*,
    float*,
    size_t,
    size_t,
    size_t,
    float*,
    float*
) {
    return false;
}

bool linearBatchForwardDevice(
    GemmDevicePath,
    const float*,
    const float*,
    const float*,
    float*,
    size_t,
    size_t,
    size_t,
    size_t
) {
    return false;
}

bool linearBatchBackwardDevice(
    GemmDevicePath,
    const float*,
    const float*,
    const float*,
    float*,
    float*,
    float*,
    float*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    float*,
    float*
) {
    return false;
}
