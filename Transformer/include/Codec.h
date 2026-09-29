//
// Created by HP on 9/29/2026.
//

#pragma once

#ifndef GPT_2_FROM_SCRATCH_CODEC_H
#define GPT_2_FROM_SCRATCH_CODEC_H

#include <string>
#include <locale>
#include <vector>
#include <map>
#include <utility>

class Codec {

    public:
    static std::string to_utf8(const std::wstring& wstring);
    static std::wstring from_utf8(const std::string& string);

    static std::vector<int> to_tokens(const std::string& string);

    static std::map<std::pair<int, int>, int> get_stats(const std::vector<int>& tokens);
    static std::vector<int> merge(const std::vector<int>& tokens, const std::pair<int, int>& pair, int idx);
};

#endif //GPT_2_FROM_SCRATCH_CODEC_H