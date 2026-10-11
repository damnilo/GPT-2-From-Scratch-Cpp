//
// Created by HP on 10/11/2026.
//

#ifndef GPT_2_FROM_SCRATCH_CHECKPOINT_H
#define GPT_2_FROM_SCRATCH_CHECKPOINT_H

#include "Model.h"
#include <string>

class Checkpoint {
public:
    static void save(const std::string& path, const Model& model);
    static Model load(const std::string& path);
};

#endif //GPT_2_FROM_SCRATCH_CHECKPOINT_H
