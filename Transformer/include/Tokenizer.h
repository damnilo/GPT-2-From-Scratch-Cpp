//
// Created by HP on 9/29/2026.
//

#ifndef GPT_2_FROM_SCRATCH_TOKENIZER_H
#define GPT_2_FROM_SCRATCH_TOKENIZER_H

#include "Codec.h"
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <utility>
#endif //GPT_2_FROM_SCRATCH_TOKENIZER_H

class Tokenizer {
    std::vector<std::pair<std::pair<int, int>, int>> merges;

    public:
    std::vector<int> encode(const std::wstring& wstring);
    void train(const std::vector<std::wstring>& texts, int num_merges);

    static std::vector<int> expand(int token, const std::map<int, std::vector<int>>& vocab);
    std::wstring decode(const std::vector<int>& vector);
};