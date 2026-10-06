//
// Created by HP on 10/4/2026.
//

#ifndef GPT_2_FROM_SCRATCH_MATH_H
#define GPT_2_FROM_SCRATCH_MATH_H
#include <cmath>
#include <limits>

class Math {

    public:
    template<typename T>
    [[nodiscard]] static float derivative(float x, T* object, float (T::*func)(float) const) {
        const float h = std::sqrt(std::numeric_limits<float>::epsilon());
        return ((object->*func)(x + h) - (object->*func)(x - h)) / (2.0f * h);
    }
};

#endif //GPT_2_FROM_SCRATCH_MATH_H
