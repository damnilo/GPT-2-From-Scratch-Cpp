//
// Created by HP on 10/14/2026.
//
#include "../NumCPP/include/Tensor.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "../NeuralNet/Layers/include/Embedding.h"
#include "../NeuralNet/Layers/include/GeLU.h"
#include "../NeuralNet/Layers/include/LayerNorm.h"
#include "../NeuralNet/Layers/include/Linear.h"
#include "../NeuralNet/Layers/include/Softmax.h"

#ifndef DATA_DIR
#define DATA_DIR "Tests/references/data"
#endif

static std::string dataPath(const std::string& fileName) {
    return std::string(DATA_DIR) + "/" + fileName;
}

static std::vector<std::string> splitLine(const std::string& line) {
    std::vector<std::string> tokens;
    for (const auto& part : line | std::views::split(' ')) {
        if (part.begin() == part.end()) {
            continue;
        }

        std::string token(part.begin(), part.end());
        if (!token.empty() && token.back() == '\r') {
            token.pop_back();
        }
        if (!token.empty()) {
            tokens.push_back(std::move(token));
        }
    }
    return tokens;
}

static std::vector<Tensor> loadData(const std::string& fileName) {
    std::ifstream ifs(dataPath(fileName));

    if (!ifs) {
        throw std::runtime_error("File could not be opened for reading: " + fileName);
    }

    std::string line;
    if (!std::getline(ifs, line)) {
        throw std::runtime_error("File is empty: " + fileName);
    }

    std::vector<Tensor> ret;
    while (std::getline(ifs, line)) {
        std::string shapeLine;
        std::string valueLine;
        if (!std::getline(ifs, shapeLine) || !std::getline(ifs, valueLine)) {
            throw std::runtime_error("Incomplete tensor record in " + fileName);
        }

        std::vector<size_t> shape;
        for (const auto& token : splitLine(shapeLine)) {
            shape.push_back(static_cast<size_t>(std::stoull(token)));
        }

        std::vector<float> values;
        for (const auto& token : splitLine(valueLine)) {
            values.push_back(std::stof(token));
        }

        ret.emplace_back(shape, values);
    }

    return ret;
}

static bool matches(const Tensor& expected, const Tensor& actual) {
    if (actual.getShape() != expected.getShape()) {
        return false;
    }

    const auto& actualData = actual.getData();
    const auto& expectedData = expected.getData();
    constexpr float atol = 1e-4f;
    constexpr float rtol = 1e-4f;

    for (size_t i = 0; i < actualData.size(); ++i) {
        const float diff = std::fabs(actualData[i] - expectedData[i]);
        const float limit = atol + rtol * std::fabs(expectedData[i]);
        if (diff > limit) {
            return false;
        }
    }

    return true;
}

static void report(const std::string& name, const Tensor& expected, const Tensor& actual, int& failures) {
    if (matches(expected, actual)) {
        std::cout << name << " correct" << std::endl;
        return;
    }

    std::cout << name << " failed" << std::endl;
    ++failures;
}

static void loadParameters(Layer& layer, const std::vector<Tensor>& values, size_t first) {
    const auto params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        *params[i] = values.at(first + i);
    }
}

int main() {
    int failures = 0;

    for (size_t i = 0; i < 4; ++i) {
        if (i == 0) {
            auto tensor = loadData("add_same.txt");
            auto tensor1 = loadData("add_broadcast_one.txt");
            auto tensor2 = loadData("add_broadcast_axis.txt");

            auto res = tensor[0] + tensor[1];
            auto res1 = tensor1[0] + tensor1[1];
            auto res2 = tensor2[0] + tensor2[1];

            report("same addition", tensor[2], res, failures);
            report("one addition", tensor1[2], res1, failures);
            report("axis addition", tensor2[2], res2, failures);
        } else if (i == 1) {
            auto tensor = loadData("sub_same.txt");
            auto tensor1 = loadData("sub_broadcast_one.txt");
            auto tensor2 = loadData("sub_broadcast_axis.txt");

            auto res = tensor[0] - tensor[1];
            auto res1 = tensor1[0] - tensor1[1];
            auto res2 = tensor2[0] - tensor2[1];

            report("same subtraction", tensor[2], res, failures);
            report("one subtraction", tensor1[2], res1, failures);
            report("axis subtraction", tensor2[2], res2, failures);
        } else if (i == 2) {
            auto tensor = loadData("mul_same.txt");
            auto tensor1 = loadData("mul_broadcast_one.txt");
            auto tensor2 = loadData("mul_broadcast_axis.txt");

            auto res = tensor[0] * tensor[1];
            auto res1 = tensor1[0] * tensor1[1];
            auto res2 = tensor2[0] * tensor2[1];

            report("same multiplication", tensor[2], res, failures);
            report("one multiplication", tensor1[2], res1, failures);
            report("axis multiplication", tensor2[2], res2, failures);
        } else {
            auto tensor = loadData("div_same.txt");
            auto tensor1 = loadData("div_broadcast_one.txt");
            auto tensor2 = loadData("div_broadcast_axis.txt");

            auto res = tensor[0] / tensor[1];
            auto res1 = tensor1[0] / tensor1[1];
            auto res2 = tensor2[0] / tensor2[1];

            report("same division", tensor[2], res, failures);
            report("one division", tensor1[2], res1, failures);
            report("axis division", tensor2[2], res2, failures);
        }
    }

    auto test = loadData("matmul_small.txt");
    auto test1 = loadData("matmul_out.txt");
    auto test2 = loadData("batch_matmul_small.txt");

    auto matmul_res = test[0].matmul(test[1]);
    auto matmul_res1 = test1[0].matmul(test1[1]);
    auto matmul_res2 = test2[0].batchMatmul(test2[1]);

    report("small matmul", test[2], matmul_res, failures);
    report("large matmul", test1[2], matmul_res1, failures);
    report("batch matmul", test2[2], matmul_res2, failures);

    auto linear = loadData("linear.txt");
    auto embedding = loadData("embedding.txt");
    auto layer_norm = loadData("layer_norm.txt");
    auto softmax = loadData("softmax.txt");
    auto gelu = loadData("gelu.txt");

    Linear l(3, 4);
    loadParameters(l, linear, 1);
    Tensor lin_res = l.forward(linear[0]);
    report("linear forward", linear[4], lin_res, failures);
    Tensor lin_grad = l.backward(linear[3]);
    report("linear backward", linear[5], lin_grad, failures);
    const auto linearGrads = l.gradients();
    report("linear weight grad", linear[6], *linearGrads[0], failures);
    report("linear bias grad", linear[7], *linearGrads[1], failures);

    Embedding e(3, 4);
    loadParameters(e, embedding, 1);
    Tensor emb_res = e.forward(embedding[0]);
    report("embedding forward", embedding[3], emb_res, failures);
    e.backward(embedding[2]);
    const auto embeddingGrads = e.gradients();
    report("embedding weight grad", embedding[4], *embeddingGrads[0], failures);

    LayerNorm ln(3);
    loadParameters(ln, layer_norm, 1);
    Tensor ln_res = ln.forward(layer_norm[0]);
    report("layer norm forward", layer_norm[4], ln_res, failures);
    Tensor ln_grad = ln.backward(layer_norm[3]);
    report("layer norm backward", layer_norm[5], ln_grad, failures);
    const auto layerNormGrads = ln.gradients();
    report("layer norm gamma grad", layer_norm[6], *layerNormGrads[0], failures);
    report("layer norm beta grad", layer_norm[7], *layerNormGrads[1], failures);

    Softmax s(static_cast<int>(softmax[1][0]));
    Tensor s_res = s.forward(softmax[0]);
    report("softmax forward", softmax[3], s_res, failures);
    Tensor s_grad = s.backward(softmax[2]);
    report("softmax backward", softmax[4], s_grad, failures);

    GeLU g{};
    Tensor g_res = g.forward(gelu[0]);
    report("gelu forward", gelu[2], g_res, failures);
    Tensor g_grad = g.backward(gelu[1]);
    report("gelu backward", gelu[3], g_grad, failures);

    return failures == 0 ? 0 : 1;
}
