#pragma once

#include <cstdint>
#include <string>

struct TrainConfig {
    size_t n_layers = 3;
    size_t embedding_dim = 32;
    size_t n_heads = 4;
    size_t block_size = 64;
    size_t batch_size = 1;
    float learning_rate = 0.001f;
    float weight_decay = 0.01f;
    size_t epochs = 10;
    std::uint64_t seed = 1;
    std::string corpus;
    std::string merges;
    std::string checkpoint;
};

TrainConfig loadConfig(const std::string& path);
void runTrain(const TrainConfig& config, bool overwrite);
void runResume(const TrainConfig& config);
void runGenerate(const TrainConfig& config, size_t tokens);
