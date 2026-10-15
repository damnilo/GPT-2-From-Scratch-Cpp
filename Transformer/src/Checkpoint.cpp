//
// Created by HP on 10/11/2026.
//

#include "../include/Checkpoint.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t magic = 0x47505432;
constexpr std::uint32_t version_weights = 1;
constexpr std::uint32_t version_optimizer = 2;
constexpr std::uint32_t version_layers = 3;

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

std::uint32_t readHeader(std::istream& in) {
    const auto file_magic = readPod<std::uint32_t>(in);
    const auto file_version = readPod<std::uint32_t>(in);

    if (file_magic != magic || (file_version != version_weights && file_version != version_optimizer && file_version != version_layers)) {
        throw std::runtime_error("Checkpoint file is not a model snapshot");
    }

    return file_version;
}

ModelConfig readConfig(std::istream& in, std::uint32_t version) {
    ModelConfig config{};
    config.vocab_size = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.block_size = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.embedding_dim = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.n_heads = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.batch_size = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.sequence_len = static_cast<size_t>(readPod<std::uint64_t>(in));
    config.n_layers = version >= version_layers ? static_cast<size_t>(readPod<std::uint64_t>(in)) : 3;
    return config;
}

void writeConfig(std::ostream& out, const ModelConfig& config, std::uint32_t version) {
    writePod<std::uint64_t>(out, config.vocab_size);
    writePod<std::uint64_t>(out, config.block_size);
    writePod<std::uint64_t>(out, config.embedding_dim);
    writePod<std::uint64_t>(out, config.n_heads);
    writePod<std::uint64_t>(out, config.batch_size);
    writePod<std::uint64_t>(out, config.sequence_len);
    if (version >= version_layers) {
        writePod<std::uint64_t>(out, config.n_layers);
    }
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

std::pair<std::vector<size_t>, std::vector<float>> readTensorData(std::istream& in) {
    const auto rank = static_cast<size_t>(readPod<std::uint64_t>(in));
    std::vector<size_t> shape(rank);

    for (size_t i = 0; i < rank; i++) {
        shape[i] = static_cast<size_t>(readPod<std::uint64_t>(in));
    }

    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }

    std::vector<float> data(count);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));

    if (!in) {
        throw std::runtime_error("Checkpoint could not be read");
    }

    return {shape, data};
}

void readTensor(std::istream& in, Tensor& tensor) {
    auto [shape, data] = readTensorData(in);

    if (shape != tensor.getShape()) {
        throw std::runtime_error("Checkpoint parameter shape does not match the model");
    }

    for (size_t i = 0; i < data.size(); i++) {
        tensor[i] = data[i];
    }
}

Tensor readNewTensor(std::istream& in) {
    auto [shape, data] = readTensorData(in);
    return {shape, data};
}

void writeWeights(std::ostream& out, const Model& model, std::uint32_t version) {
    writeConfig(out, model.getConfig(), version);

    const auto parameters = model.parameters();
    writePod<std::uint64_t>(out, parameters.size());

    for (const Tensor* parameter : parameters) {
        writeTensor(out, *parameter);
    }
}

Model readWeights(std::istream& in, std::uint32_t version) {
    const ModelConfig config = readConfig(in, version);
    Model model(config.vocab_size, config.block_size, config.embedding_dim, config.n_heads, config.n_layers, config.batch_size, config.sequence_len);

    const auto count = readPod<std::uint64_t>(in);
    auto parameters = model.parameters();

    if (count != parameters.size()) {
        throw std::runtime_error("Checkpoint parameter count does not match the model");
    }

    for (Tensor* parameter : parameters) {
        readTensor(in, *parameter);
    }

    return model;
}

void writeOptimizer(std::ostream& out, const AdamW& optimizer, std::uint64_t seed, std::uint64_t epoch, std::uint64_t step) {
    const auto& moments = optimizer.getMoments();
    const auto& velocities = optimizer.getVelocities();

    if (moments.size() != velocities.size()) {
        throw std::runtime_error("Checkpoint AdamW moments and velocities differ in count");
    }

    writePod<std::uint64_t>(out, optimizer.getTimestep());
    writePod<std::uint64_t>(out, epoch);
    writePod<std::uint64_t>(out, step);
    writePod<std::uint64_t>(out, seed);
    writePod<std::uint64_t>(out, moments.size());

    for (size_t i = 0; i < moments.size(); i++) {
        writeTensor(out, moments[i]);
        writeTensor(out, velocities[i]);
    }
}

void readOptimizer(std::istream& in, const Model& model, AdamW& optimizer, std::uint64_t& seed, std::uint64_t& epoch, std::uint64_t& step) {
    const auto timestep = static_cast<size_t>(readPod<std::uint64_t>(in));
    epoch = readPod<std::uint64_t>(in);
    step = readPod<std::uint64_t>(in);
    seed = readPod<std::uint64_t>(in);

    const auto count = static_cast<size_t>(readPod<std::uint64_t>(in));
    std::vector<Tensor> moments;
    std::vector<Tensor> velocities;
    moments.reserve(count);
    velocities.reserve(count);

    for (size_t i = 0; i < count; i++) {
        moments.push_back(readNewTensor(in));
        velocities.push_back(readNewTensor(in));
    }

    const auto parameters = model.parameters();
    if (!moments.empty() && moments.size() != parameters.size()) {
        throw std::runtime_error("Checkpoint AdamW state does not match the model");
    }

    for (size_t i = 0; i < moments.size(); i++) {
        if (moments[i].getShape() != parameters[i]->getShape()) {
            throw std::runtime_error("Checkpoint AdamW moment shape does not match the model");
        }
    }

    optimizer.restore(timestep, std::move(moments), std::move(velocities));
}

std::ofstream openWrite(const std::string& path) {
    std::ofstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Checkpoint could not be opened for writing");
    }

    return file;
}

std::ifstream openRead(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Checkpoint could not be opened for reading");
    }

    return file;
}

}

void Checkpoint::save(const std::string& path, const Model& model) {
    std::ofstream file = openWrite(path);

    writePod(file, magic);
    writePod(file, version_layers);
    writeWeights(file, model, version_layers);
    writePod<std::uint64_t>(file, 0);

    if (!file) {
        throw std::runtime_error("Checkpoint could not be written");
    }
}

void Checkpoint::save(const std::string& path, const Model& model, const AdamW& optimizer, std::uint64_t seed, std::uint64_t epoch, std::uint64_t step) {
    std::ofstream file = openWrite(path);

    writePod(file, magic);
    writePod(file, version_layers);
    writeWeights(file, model, version_layers);
    writePod<std::uint64_t>(file, 1);
    writeOptimizer(file, optimizer, seed, epoch, step);

    if (!file) {
        throw std::runtime_error("Checkpoint could not be written");
    }
}

Model Checkpoint::load(const std::string& path) {
    std::ifstream file = openRead(path);
    const std::uint32_t version = readHeader(file);
    return readWeights(file, version);
}

Model Checkpoint::load(const std::string& path, AdamW& optimizer, std::uint64_t& seed, std::uint64_t& epoch, std::uint64_t& step) {
    std::ifstream file = openRead(path);
    const std::uint32_t version = readHeader(file);
    Model model = readWeights(file, version);

    if (version == version_weights) {
        throw std::runtime_error("Checkpoint version 1 stores weights only");
    }

    if (version >= version_layers) {
        const auto has_optimizer = readPod<std::uint64_t>(file);
        if (has_optimizer == 0) {
            throw std::runtime_error("Checkpoint stores weights only");
        }
    }

    readOptimizer(file, model, optimizer, seed, epoch, step);
    return model;
}
