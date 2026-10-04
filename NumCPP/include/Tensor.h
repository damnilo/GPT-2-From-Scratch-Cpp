//
// Created by HP on 10/1/2026.
//

#ifndef GPT_2_FROM_SCRATCH_TENSOR_H
#define GPT_2_FROM_SCRATCH_TENSOR_H
#include <vector>
#include <initializer_list>
#include <random>

class Tensor {
    private:
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    [[nodiscard]] size_t calculateSize() const;
    void calculateStrides();
    static std::mt19937& generator();

    public:

    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float value);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
    Tensor(std::initializer_list<size_t> shape, std::initializer_list<float> data);

    [[nodiscard]] const std::vector<size_t>& getShape() const;
    [[nodiscard]] const std::vector<size_t>& getStrides() const;
    [[nodiscard]] const std::vector<float>& getData() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t ndim() const;
    [[nodiscard]] bool empty() const;

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    float& at(const std::vector<size_t>& index);
    [[nodiscard]] const float& at(const std::vector<size_t>& index) const;

    void fill(float value);
    void zeros();
    void ones();
    void randomize(float min = -1.0f, float max = 1.0f);

    [[nodiscard]] Tensor reshape(const std::vector<size_t>& shape) const;
    [[nodiscard]] Tensor transpose() const;
    [[nodiscard]] Tensor flatten() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float value) const;
    Tensor operator-(float value) const;
    Tensor operator*(float value) const;
    Tensor operator/(float value) const;

    [[nodiscard]] Tensor matmul(const Tensor& other) const;
    [[nodiscard]] Tensor dot(const Tensor& other) const;

    [[nodiscard]] Tensor exp() const;
    [[nodiscard]] Tensor log() const;
    [[nodiscard]] Tensor sqrt() const;
    [[nodiscard]] Tensor pow(float value) const;

    [[nodiscard]] Tensor sum() const;
    [[nodiscard]] Tensor sum(int axis) const;

    [[nodiscard]] Tensor min() const;

    [[nodiscard]] Tensor max() const;
    [[nodiscard]] Tensor max(int axis) const;

    [[nodiscard]] Tensor mean() const;
    [[nodiscard]] Tensor mean(int axis) const;

    void print() const;
};

#endif //GPT_2_FROM_SCRATCH_TENSOR_H
