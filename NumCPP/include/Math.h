//
// Created by HP on 10/4/2026.
//

#ifndef GPT_2_FROM_SCRATCH_MATH_H
#define GPT_2_FROM_SCRATCH_MATH_H
#include <cmath>
#include <limits>
#include "../../NeuralNet/include/GeLU.h"

class Math {

    public:
    template<typename T>
    [[nodiscard]] static float derivative(float x, T* object, float (T::*func)(float) const);
};

#endif //GPT_2_FROM_SCRATCH_MATH_H
