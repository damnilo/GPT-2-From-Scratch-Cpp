//
// Created by HP on 9/29/2026.
//

#include "../include/Tokenizer.h"
#include <algorithm>

std::vector<int> Tokenizer::tokenize(const std::wstring& utf_string) {
    int next_bytes = 256;

    std::string string = codec.to_utf8(utf_string);
    std::vector<int> bytes = codec.to_bytes(string);

    while (true) {
        std::map<std::pair<int, int>, int> map = codec.get_stats(bytes);
        if (map.empty()) break;

        auto max_it = std::max_element(map.begin(), map.end(),
            [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        auto best_pair = max_it->first;
        int max_count = max_it->second;

        if (max_count < 2) break;

        bytes = codec.merge(bytes, best_pair, next_bytes);
        ++next_bytes;
    }

    return bytes;
}