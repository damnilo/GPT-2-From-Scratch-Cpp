#pragma once

#include <cstddef>

bool rooflineSoftmax(float* scores, size_t H, size_t T, float scale);

bool rooflineLayerNorm(float* x, float* y, float* gamma, float* beta, size_t T, size_t C);
