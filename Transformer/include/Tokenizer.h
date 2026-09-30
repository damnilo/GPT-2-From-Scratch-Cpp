//
// Created by HP on 9/29/2026.
//

#pragma once

#ifndef GPT_2_FROM_SCRATCH_TOKENIZER_H
#define GPT_2_FROM_SCRATCH_TOKENIZER_H

#include "Codec.h"
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <regex>

class Tokenizer {
    std::wregex gpt4_split_pattern = std::wregex(LR"('(?:[sStTmMdD]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]{1,3}| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]|\s+)");
    std::vector<std::pair<std::pair<int, int>, int>> merges;

    public:
    std::vector<int> encode(const std::wstring& wstring);
    void train(const std::vector<std::wstring>& texts, int num_merges);

    static std::vector<int> expand(int token, const std::map<int, std::vector<int>>& vocab);
    std::wstring decode(const std::vector<int>& vector);

    std::vector<std::wstring> split(const std::wstring& text);

    std::vector<std::pair<std::pair<int, int>, int>> getMerges();
};

#endif //GPT_2_FROM_SCRATCH_TOKENIZER_H