//
// Created by HP on 10/11/2026.
//

#ifndef GPT_2_FROM_SCRATCH_CHECKPOINT_H
#define GPT_2_FROM_SCRATCH_CHECKPOINT_H

#include "Model.h"
#include "../../NeuralNet/Optimizers/include/AdamW.h"

#include <cstdint>
#include <string>

class Checkpoint {
public:
    static void save(const std::string& path, const Model& model);
    static void save(const std::string& path, const Model& model, const AdamW& optimizer, std::uint64_t seed, std::uint64_t epoch, std::uint64_t step);
    static Model load(const std::string& path);
    static Model load(const std::string& path, AdamW& optimizer, std::uint64_t& seed, std::uint64_t& epoch, std::uint64_t& step);
};

#endif //GPT_2_FROM_SCRATCH_CHECKPOINT_H
