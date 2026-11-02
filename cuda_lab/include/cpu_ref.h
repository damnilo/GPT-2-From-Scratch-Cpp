#pragma once

#include <cstddef>
#include <random>
#include <vector>

inline void fill_uniform(std::vector<float>& vec, unsigned seed = 1 {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : vec) {
        v = dist(rng);
    }
}

inline void cpu_gemm(const float* a, const float* b, float* c, 
                    size_t rows, size_t common, size_t cols) {
    
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            float sum = 0.0f;

            for (size_t k = 0; k < common; ++k) {
                sum += a[row * common + k] * b[k * cols + col];
            }

            c[row * cols + col] = sum;
        }
    }
}
