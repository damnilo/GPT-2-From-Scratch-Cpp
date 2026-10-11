//
// Created by HP on 10/11/2026.
//

#include "../include/Checkpoint.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t magic = 0x47505432;
constexpr std::uint32_t version = 1;

template<typename T>
void writePod(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
T readPod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));

    if (!in) {
        throw std::runtime_error("Checkpoint could not be read");
    }

    return value;
}

void checkHeader(std::istream& in) {
    const auto file_magic = readPod<std::uint32_t>(in);
    const auto file_version = readPod<std::uint32_t>(in);

    if (file_magic != magic || file_version != version) {
        throw std::runtime_error("Checkpoint file is not a model snapshot");
    }
}

ModelConfig readConfig(std::istream& in) {
    ModelConfig config{};
    config.vocab_size = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.block_size = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.embedding_dim = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.n_heads = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.batch_size = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.sequence_len = static_cast<size_t>(readPod<std::uint64_t>(in));
    return config;
}

void writeConfig(std::ostream& out, const ModelConfig& config) {
    writePod<std::uint64_t>(out, config.vocab_size);
    writePod<std::uint64_t>(out, config.block_size);
    writePod<std::uint64_t>(out, config.embedding_dim);
    writePod<std::uint64_t>(out, config.n_heads);
    writePod<std::uint64_t>(out, config.batch_size);
    writePod<std::uint64_t>(out, config.sequence_len);
}

void writeTensor(std::ostream& out, const Tensor& tensor) {
    const auto& shape = tensor.getShape();
    writePod<std::uint64_t>(out, shape.size());

    for (size_t dim : shape) {
        writePod<std::uint64_t>(out, dim);
    }

    const auto& data = tensor.getData();
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

void readTensor(std::istream& in, Tensor& tensor) {
    const auto rank = static_cast<size_t>(readPod<std::uint64_t>(in));
    std::vector<size_t> shape(rank);

    for (size_t i = 0; i < rank; i++) {
        shape[i] = static_cast<size_t>(readPod<std::uint64_t>(in));
    }

    if (shape != tensor.getShape()) {
        throw std::runtime_error("Checkpoint parameter shape does not match the model");
    }

    std::vector<float> data(tensor.size());
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));

    if (!in) {
        throw std::runtime_error("Checkpoint could not be read");
    }

    for (size_t i = 0; i < data.size(); i++) {
        tensor[i] = data[i];
    }
}

}

void Checkpoint::save(const std::string& path, const Model& model) {
    std::ofstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Checkpoint could not be opened for writing");
    }

    writePod(file, magic);
    writePod(file, version);
    writeConfig(file, model.getConfig());

    const auto parameters = model.parameters();
    writePod<std::uint64_t>(file, parameters.size());

    for (const Tensor* parameter : parameters) {
        writeTensor(file, *parameter);
    }

    if (!file) {
        throw std::runtime_error("Checkpoint could not be written");
    }
}

Model Checkpoint::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Checkpoint could not be opened for reading");
    }

    checkHeader(file);
    const ModelConfig config = readConfig(file);

    Model model(config.vocab_size, config.block_size, config.embedding_dim, config.n_heads, config.batch_size, config.sequence_len);

    const auto count = readPod<std::uint64_t>(file);
    auto parameters = model.parameters();

    if (count != parameters.size()) {
        throw std::runtime_error("Checkpoint parameter count does not match the model");
    }

    for (Tensor* parameter : parameters) {
        readTensor(file, *parameter);
    }

    return model;
}
