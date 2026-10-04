//
// Created by HP on 10/1/2026.
//

#include "../include/Tensor.h"

#include <stdexcept>
#include <random>
#include <cmath>
#include <iostream>

std::mt19937& Tensor::generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    return gen;
};

size_t Tensor::calculateSize() const {
    size_t size = 1;

    for (size_t dim : shape) {
        size *= dim;
    }

    return size;
}

void Tensor::calculateStrides() {
    this->strides.resize(shape.size());

    size_t stride = 1;

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        this->strides[i] = stride;
        stride *= shape[i];
    }
}

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<size_t> &shape) {
    this->shape = shape;
    calculateStrides();
    this->data.resize(calculateSize(), 0.0f);
}

Tensor::Tensor(const std::vector<size_t> &shape, float value) {
    this->shape = shape;
    calculateStrides();
    this->data.resize(calculateSize(), value);
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data) {
    this->shape = shape;
    calculateStrides();

    if (calculateSize() != data.size()) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    this->data = data;
}

Tensor::Tensor(std::initializer_list<size_t> shape, std::initializer_list<float> data) {
    this->shape = shape;
    calculateStrides();

    if (calculateSize() != data.size()) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    this->data = data;
}

const std::vector<size_t> & Tensor::getShape() const {
    return this->shape;
}

const std::vector<size_t> & Tensor::getStrides() const {
    return this->strides;
}

const std::vector<float> & Tensor::getData() const {
    return this->data;
}

size_t Tensor::size() const {
    return calculateSize();
}

size_t Tensor::ndim() const {
    return shape.size();
}

bool Tensor::empty() const {
    return data.empty();
}

float & Tensor::operator[](size_t index) {
    return data[index];
}

const float & Tensor::operator[](size_t index) const {
    return data[index];
}

float & Tensor::at(const std::vector<size_t> &index) {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    size_t lin_index = 0;

    for (size_t i = 0; i < index.size(); i++) {
        if (index[i] >= shape[i]) {
            throw std::out_of_range("The index is out of bounds.");
        }

        lin_index += index[i] * strides[i];
    }

    return data[lin_index];
}

const float & Tensor::at(const std::vector<size_t> &index) const {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    size_t lin_index = 0;

    for (int i = 0; i < index.size(); i++) {
        if (index[i] >= shape[i]) {
            throw std::invalid_argument("The index is out of bounds.");
        }

        lin_index += index[i] * strides[i];
    }

    return data[lin_index];
}

void Tensor::fill(float value) {
    for (float& i : this->data) {
        i = value;
    }
}

void Tensor::zeros() {
    for (float& i : this->data) {
        i = 0.0f;
    }
}

void Tensor::ones() {
    for (float& i : this->data) {
        i = 1.0f;
    }
}

void Tensor::randomize(float min, float max) {
    auto& gen = generator();
    std::uniform_real_distribution<float> distribution(min, max);

    for (float& i : this->data) {
        i = distribution(gen);
    }
}

Tensor Tensor::reshape(const std::vector<size_t> &newShape) const {
    size_t newSize = 1;

    for (size_t dim : newShape) {
        newSize *= dim;
    }

    if (newSize != data.size()) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    Tensor result(newShape, data);

    return result;
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be 2 for transposing.");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    Tensor result({cols, rows});

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.at({j, i}) = this->at({i, j});
        }
    }

    return result;
}

Tensor Tensor::flatten() const {
    Tensor result({calculateSize()}, this->data);

    return result;
}

Tensor Tensor::operator+(const Tensor &other) const {
    if (this->shape == other.shape) {
        Tensor newTensor = Tensor(this->shape, this->data);

        for (size_t i = 0; i < this->data.size(); i++) {
            newTensor.data[i] = this->data[i] + other.data[i];
        }

        return newTensor;
    }

    if (this->shape.size() == 2 &&
        other.shape.size() == 1 &&
        this->shape[1] == other.shape[0]) {
        size_t rows = this->shape[0];
        size_t cols = this->shape[1];

        Tensor res(this->shape, this->data);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                res[i * cols + j] += other.data[j];
            }
        }

        return res;
    }

    throw std::invalid_argument("Incompatible shapes for addition");
}

Tensor Tensor::operator-(const Tensor &other) const {
    if (this->shape == other.shape) {
        Tensor newTensor = Tensor(this->shape, this->data);

        for (size_t i = 0; i < this->data.size(); i++) {
            newTensor.data[i] = this->data[i] - other.data[i];
        }

        return newTensor;
    }

    if (this->shape.size() == 2 &&
        other.shape.size() == 1 &&
        this->shape[1] == other.shape[0]) {
        size_t rows = this->shape[0];
        size_t cols = this->shape[1];

        Tensor res(this->shape, this->data);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                res[i * cols + j] -= other.data[j];
            }
        }

        return res;
        }

    throw std::invalid_argument("Incompatible shapes for addition");
}

Tensor Tensor::operator*(const Tensor &other) const {
    if (this->shape != other.shape) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    Tensor newTensor = Tensor(this->shape, this->data);

    for (size_t i = 0; i < this->data.size(); i++) {
        newTensor.data[i] = this->data[i] * other.data[i];
    }

    return newTensor;
}

Tensor Tensor::operator/(const Tensor &other) const {
    if (this->shape != other.shape) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    Tensor newTensor = Tensor(this->shape, this->data);

    for (size_t i = 0; i < this->data.size(); i++) {
        if (other.data[i] == 0.0f) {
            throw std::invalid_argument("The value cannot be zero.");
        }

        newTensor.data[i] = this->data[i] / other.data[i];
    }

    return newTensor;
}

Tensor Tensor::operator+(float value) const {
    Tensor newTensor = Tensor(this->shape, this->data);

    for (size_t i = 0; i < this->data.size(); i++) {
        newTensor.data[i] = this->data[i] + value;
    }

    return newTensor;
}

Tensor Tensor::operator-(float value) const {
    Tensor newTensor = Tensor(this->shape, this->data);

    for (size_t i = 0; i < this->data.size(); i++) {
        newTensor.data[i] = this->data[i] - value;
    }

    return newTensor;
}

Tensor Tensor::operator*(float value) const {
    Tensor newTensor = Tensor(this->shape, this->data);

    for (size_t i = 0; i < this->data.size(); i++) {
        newTensor.data[i] = this->data[i] * value;
    }

    return newTensor;
}

Tensor Tensor::operator/(float value) const {
    if (value == 0.0f) {
        throw std::invalid_argument("The value cannot be zero.");
    }

    Tensor newTensor = Tensor(this->shape, this->data);

    for (size_t i = 0; i < this->data.size(); i++) {
        newTensor.data[i] = this->data[i] / value;
    }

    return newTensor;
}

Tensor Tensor::matmul(const Tensor &other) const {
    if (this->shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be 2.");
    }

    if (this->shape[1] != other.shape[0]) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    size_t rows = this->shape[0];
    size_t common = this->shape[1];
    size_t cols = other.shape[1];

    Tensor result({rows, cols}, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(rows); i++) {
        for (size_t j = 0; j < cols; j++) {
            float sum = 0.0f;

            for (size_t k = 0; k < common; k++) {
                sum += this->data[i * common + k] * other.data[k * cols + j];
            }

            result[i * cols + j] = sum;
        }
    }

    return result;
}

Tensor Tensor::dot(const Tensor &other) const {
    if (this->shape.size() != 1 || other.shape.size() != 1) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be 1.");
    }

    if (this->shape != other.shape) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be the same.");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < this->shape[0]; i++) {
        sum += this->data[i] * other.data[i];
    }

    return Tensor({}, {sum});
}

Tensor Tensor::exp() const {
    std::vector<float> ret(data.size());

    for (size_t i = 0; i < this->data.size(); i++) {
        ret[i] = std::exp(data[i]);
    }

    return {this->shape, ret};
}

Tensor Tensor::log() const {
    std::vector<float> ret(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        ret[i] = std::log(data[i]);
    }

    return {this->shape, ret};
}

Tensor Tensor::sqrt() const {
    std::vector<float> ret(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        ret[i] = std::sqrt(data[i]);
    }

    return {this->shape, ret};
}

Tensor Tensor::pow(float value) const {
    std::vector<float> ret(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        ret[i] = std::pow(data[i], value);
    }

    return {this->shape, ret};
}

Tensor Tensor::sum() const {
    float sum = 0.0f;

    for (float i : this->data) {
        sum += i;
    }

    return {{}, {sum}};
}

Tensor Tensor::sum(int axis) const {
    if (shape.empty()) {
        throw std::invalid_argument("Softmax cannot be applied to a scalar.");
    }

    if (data.empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }

    if (axis == -1) {
        axis = static_cast<int>(shape.size()) - 1;
    }

    if (axis < 0 || axis >= shape.size()) {
        throw std::invalid_argument("The axis must be greater than 0 or -1.");
    }

    std::vector<size_t> res_shape;

    for (size_t i = 0; i < shape.size(); i++) {
        if (i != axis) {
            res_shape.push_back(shape[i]);
        }
    }

    if (res_shape.empty()) {
        return sum();
    }

    Tensor ret(res_shape, 0.0f);

    size_t out_size = 1;
    const size_t axis_size = shape[axis];

    for (int i = 0; i < axis; i++) {
        out_size *= shape[i];
    }

    size_t inner_size = 1;
    for (size_t i = axis+1; i < shape.size(); i++) {
        inner_size *= shape[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(out_size); i++) {
        for (size_t k = 0; k < inner_size; k++) {
            size_t group_start = i * axis_size * inner_size + k;

            float sum = 0.0f;

            for (size_t j = 0; j < axis_size; j++) {
                size_t idx = group_start + j * inner_size;

                sum += data[idx];
            }

            size_t out_idx = i * inner_size + k;
            ret[out_idx] = sum;
        }
    }

    return ret;
}

Tensor Tensor::min() const {
    if (data.empty()) {
        throw std::invalid_argument("The data cannot be one dimensional.");
    }
    float min = std::numeric_limits<float>::max();

    for (float i : this->data) {
        min = std::min(i, min);
    }

    return {{}, {min}};
}

Tensor Tensor::max() const {
    if (data.empty()) {
        throw std::invalid_argument("The data cannot be one dimensional.");
    }

    float max = std::numeric_limits<float>::lowest();

    for (float i : this->data) {
        max = std::max(i, max);
    }

    return {{}, {max}};
}

Tensor Tensor::max(int axis) const {
    if (data.empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }

    if (shape.empty()) {
        throw std::invalid_argument("The shape cannot be empty.");
    }

    if (axis == -1) axis = static_cast<int>(shape.size()) - 1;

    if (axis < 0 || axis >= shape.size()) {
        throw std::invalid_argument("The axis must be greater than 0 or -1.");
    }

    std::vector<size_t> res_shape;

    for (size_t i = 0; i < shape.size(); i++) {
        if (i != axis) {
            res_shape.push_back(shape[i]);
        }
    }

    if (res_shape.empty()) {
        return max();
    }

    auto axis_size = shape[axis];
    size_t outer_size = 1;
    for (size_t i = 0; i < axis; i++) {
        outer_size *= shape[i];
    }

    size_t inner_size = 1;
    for (size_t i = axis+1; i < shape.size(); i++) {
        inner_size *= shape[i];
    }

    Tensor ret(res_shape, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(outer_size); i++) {
        for (size_t k = 0; k < inner_size; k++) {
            float max_value = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < axis_size; j++) {
                if (max_value < data[i * axis_size * inner_size + j * inner_size + k]) {
                    max_value = data[i * axis_size * inner_size + j * inner_size + k];
                }
            }

            ret[i * inner_size + k] = max_value;
        }
    }

    return ret;
}

Tensor Tensor::mean() const {
    if (data.empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }
    float mean = 0.0;

    for (float i : this->data) {
        mean += i;
    }

    return {{}, {mean/static_cast<float>(this->data.size())}};
}

Tensor Tensor::mean(int axis) const {
    if (data.empty()) {
        throw std::invalid_argument("The data cannot be empty.");
    }

    if (axis == -1) axis = static_cast<int>(shape.size()) - 1;

    if (axis < 0 || axis >= shape.size()) {
        throw std::invalid_argument("The axis must be greater than 0.");
    }

    Tensor ret = sum(axis);

    return ret / static_cast<float>(shape[axis]);
}

void Tensor::print() const {
    std::cout << "Shape: [";

    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];

        if (i + 1 < shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]" << std::endl;
    std::cout << "Data: [";

    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i];

        if (i + 1 < data.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]" << std::endl;
}
