//
// Created by HP on 10/1/2026.
//

#include "../include/Tensor.h"

#include <stdexcept>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

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

void Tensor::lower_triangular() {
    if (shape.size() == 3) {

        if (shape[1] != shape[2]) throw std::invalid_argument("3D Tensor must have its 2nd and 3rd dimension the same");

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(shape[0]); i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                for (size_t k = j+1; k < shape[2]; k++) {
                    data[i * shape[1] * shape[2] + j * shape[2] + k] = 0.0f;
                }
            }
        }

    }else if (shape.size() == 2) {

        if (shape[0] != shape[1]) throw std::invalid_argument("2D Tensor must be a square matrix");

        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = i+1; j < shape[1]; j++) {
                data[i * shape[1] + j] = 0.0f;
            }
        }

    }else {
        throw std::invalid_argument("Tensor must have 2 or 3 dimensions");
    }
}

void Tensor::upper_triangular() {
    if (shape.size() == 3) {

        if (shape[1] != shape[2]) throw std::invalid_argument("3D Tensor must have its 2nd and 3rd dimension the same");

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(shape[0]); i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                for (size_t k = 0; k < j; k++) {
                    data[i * shape[1] * shape[2] + j * shape[2] + k] = 0.0f;
                }
            }
        }

    }else if (shape.size() == 2) {

        if (shape[0] != shape[1]) throw std::invalid_argument("2D Tensor must be a square matrix");

        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < i; j++) {
                data[i * shape[1] + j] = 0.0f;
            }
        }

    }else {
        throw std::invalid_argument("Tensor must have 2 or 3 dimensions");
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

Tensor Tensor::transpose(int axis1, int axis2) const {
    if (shape.size() != 3) {
        throw std::invalid_argument("Tensor dimension must be 3");
    }

    if (axis1 == axis2) {
        throw std::invalid_argument("axis1 and axis2 cannot be same");
    }

    if (axis1 == -1) axis1 = static_cast<int>(shape.size() - 1);
    if (axis2 == -1) axis2 = static_cast<int>(shape.size() - 1);

    size_t rows = shape[axis1];
    size_t cols = shape[axis2];

    Tensor ret(shape);

    std::vector<size_t> input_index(3);
    std::vector<size_t> output_index(3);

    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {

                input_index = {i, j, k};
                output_index = input_index;

                output_index[axis1] = input_index[axis2];
                output_index[axis2] = input_index[axis1];

                ret.at(output_index) = this->at(input_index);
            }
        }
    }

    return ret;
}

Tensor Tensor::flatten() const {
    Tensor result({calculateSize()}, this->data);

    return result;
}

namespace {

std::vector<size_t> broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    const size_t rank = std::max(a.size(), b.size());
    std::vector<size_t> out(rank);

    for (size_t i = 0; i < rank; i++) {
        const size_t a_dim = i < a.size() ? a[a.size() - 1 - i] : 1;
        const size_t b_dim = i < b.size() ? b[b.size() - 1 - i] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }

        out[rank - 1 - i] = std::max(a_dim, b_dim);
    }

    return out;
}

size_t broadcastOffset(const std::vector<size_t>& coords,
                       const std::vector<size_t>& shape,
                       const std::vector<size_t>& strides) {
    if (shape.empty()) {
        return 0;
    }

    const size_t shift = coords.size() - shape.size();
    size_t index = 0;

    for (size_t i = 0; i < shape.size(); i++) {
        const size_t coord = shape[i] == 1 ? 0 : coords[shift + i];
        index += coord * strides[i];
    }

    return index;
}

template<typename Op>
Tensor broadcastBinary(const Tensor& a, const Tensor& b, Op op) {
    if (a.getShape() == b.getShape()) {
        Tensor out(a.getShape(), a.getData());

        for (size_t i = 0; i < a.getData().size(); i++) {
            out[i] = op(a.getData()[i], b.getData()[i]);
        }

        return out;
    }

    const std::vector<size_t> out_shape = broadcastShape(a.getShape(), b.getShape());
    Tensor out(out_shape, 0.0f);
    std::vector<size_t> coords(out_shape.size());

    for (size_t linear = 0; linear < out.size(); linear++) {
        size_t remainder = linear;

        for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
            const size_t dim = out_shape[static_cast<size_t>(i)];
            coords[static_cast<size_t>(i)] = dim == 0 ? 0 : remainder % dim;
            remainder = dim == 0 ? 0 : remainder / dim;
        }

        const size_t ia = broadcastOffset(coords, a.getShape(), a.getStrides());
        const size_t ib = broadcastOffset(coords, b.getShape(), b.getStrides());
        out[linear] = op(a.getData()[ia], b.getData()[ib]);
    }

    return out;
}

}

Tensor Tensor::operator+(const Tensor &other) const {
    return broadcastBinary(*this, other, [](float x, float y) { return x + y; });
}

Tensor Tensor::operator-(const Tensor &other) const {
    return broadcastBinary(*this, other, [](float x, float y) { return x - y; });
}

Tensor Tensor::operator*(const Tensor &other) const {
    return broadcastBinary(*this, other, [](float x, float y) { return x * y; });
}

Tensor Tensor::operator/(const Tensor &other) const {
    return broadcastBinary(*this, other, [](float x, float y) {
        if (y == 0.0f) {
            throw std::invalid_argument("The value cannot be zero.");
        }

        return x / y;
    });
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

Tensor Tensor::batchMatmul(const Tensor &other) const {
    if (this->shape.size() != 3 || other.shape.size() != 3) {
        throw std::invalid_argument("The number of dimensions of a Tensor must be 3.");
    }

    if (this->shape[2] != other.shape[1]) {
        throw std::invalid_argument("Multiplying dimensions must be the same");
    }

    if (this->shape[0] != other.shape[0]) {
        throw std::invalid_argument("Batch size must be the same");
    }

    size_t batch_size = this->shape[0];
    size_t rows = this->shape[1];
    size_t common = this->shape[2];
    size_t cols = other.shape[2];
    Tensor ret({batch_size, rows, cols}, 0.0f);

    for (size_t i = 0; i < batch_size; i++) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < static_cast<int>(rows); j++) {
            for (size_t k = 0; k < cols; k++) {
                float sum = 0.0f;

                for (size_t l = 0; l < common; l++) {
                    sum += this->data[(i * rows * common) + j * common + l] * other.data[(i * cols * common) + l * cols + k];
                }

                ret[(i * rows * cols) + j * cols + k] = sum;
            }
        }
    }

    return ret;
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

Tensor Tensor::sum(int axis, bool keepdims) const {
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
        if (i == static_cast<size_t>(axis)) {
            if (keepdims) {
                res_shape.push_back(1);
            }
        } else {
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

Tensor Tensor::max(int axis, bool keepdims) const {
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
        if (i == static_cast<size_t>(axis)) {
            if (keepdims) {
                res_shape.push_back(1);
            }
        } else {
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
