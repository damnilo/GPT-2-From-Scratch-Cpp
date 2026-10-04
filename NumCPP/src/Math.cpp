//
// Created by HP on 10/4/2026.
//

#include "../include/Math.h"

template<typename T>
float Math::derivative(float x, T* object, float (T::*func)(float) const){
    const float h = std::sqrt(std::numeric_limits<float>::epsilon());
    return (func(x + h) - func(x - h)) / (2.0f * h);
}
