//
// Created by HP on 9/29/2026.
//

#include "../include/Tokenizer.h"

std::vector<int> Tokenizer::encode(const std::wstring& utf_string) {
    int next_bytes = 256;

    std::string string = codec.to_utf8(utf_string);
    std::vector<int> bytes = codec.to_bytes(string);

    while (vocab_size > next_bytes) {
        std::map<std::pair<int, int>, int> map = codec.get_stats(bytes);
        if (map.empty()) break;

        auto max_it = std::ranges::max_element(map.begin(), map.end(),
                                               [](const auto& a, const auto& b) {
                                                   return a.second < b.second;
                                               });
        auto best_pair = max_it->first;
        int max_count = max_it->second;

        if (max_count < 2) break;

        bytes = codec.merge(bytes, best_pair, next_bytes);
        merges[best_pair] = max_count;
        ++next_bytes;
    }

    return bytes;
}

std::vector<int> Tokenizer::expand(int token, const std::map<int, std::vector<int>>& map) {
    if (token < 256) return {token};

    std::vector<int> ret;

    for (int subtoken : map.at(token)) {
        auto expanded = expand(subtoken, map);
        ret.insert(ret.end(), expanded.begin(), expanded.end());
    }

    return ret;
}

std::wstring Tokenizer::decode(const std::vector<int>& utf_vector) {
    std::map<int, std::vector<int>> map;
    std::string string;

    for (int i = 0; i < 256; i++) {
        map[i] = {i};
    }

    for (auto& [pair, count] : merges) {
        map[count] = {pair.first, pair.second};
    }

    for (int i : utf_vector) {
        auto bytes = expand(i, map);

        for (int byte : bytes) {
            string.push_back(static_cast<char>(byte));
        }
    }

    return codec.from_utf8(string);
}