//
// Created by HP on 9/29/2026.
//

#include "../include/Tokenizer.h"

std::vector<int> Tokenizer::encode(const std::wstring& wstring) {
    std::string string = Codec::to_utf8(wstring);
    std::vector<int> tokens = Codec::to_tokens(string);

    for (const auto& [pair, count] : merges) {
        tokens = Codec::merge(tokens, pair, count);
    }

    return tokens;
}

void Tokenizer::train(const std::vector<std::wstring>& texts, int num_merges) {
    int next_bytes = 256;

    std::vector<std::vector<int>> corpus;

    for (const auto& t : texts) {
        std::string utf8 = Codec::to_utf8(t);
        corpus.push_back(Codec::to_tokens(utf8));
    }

    for (int i = 0; i < num_merges; ++i) {

        std::map<std::pair<int, int>, int> stats;

        for (const auto& bytes : corpus) {
            auto map = Codec::get_stats(bytes);

            for (const auto& [pair, count] : map) {
                stats[pair] += count;
            }
        }

        if (stats.empty()) break;

        auto max_it = std::ranges::max_element(stats.begin(), stats.end(),
                                               [](const auto& a, const auto& b) {
                                                   return a.second < b.second;
                                               });
        auto best_pair = max_it->first;
        int max_count = max_it->second;

        if (max_count < 2) break;

        for (auto& tokens : corpus) {
            tokens = Codec::merge(tokens, best_pair, next_bytes);
        }
        merges.emplace_back(best_pair, next_bytes);
        ++next_bytes;
    }
}

std::vector<int> Tokenizer::expand(int token, const std::map<int, std::vector<int>>& vocab) {
    if (token < 256) return {token};

    std::vector<int> ret;

    for (int subtoken : vocab.at(token)) {
        auto expanded = expand(subtoken, vocab);
        ret.insert(ret.end(), expanded.begin(), expanded.end());
    }

    return ret;
}

std::wstring Tokenizer::decode(const std::vector<int>& vector) {
    std::map<int, std::vector<int>> vocab;
    std::string string;

    for (int i = 0; i < 256; i++) {
        vocab[i] = {i};
    }

    for (auto& [pair, count] : merges) {
        vocab[count] = {pair.first, pair.second};
    }

    for (int i : vector) {
        auto tokens = expand(i, vocab);

        for (int token : tokens) {
            string.push_back(static_cast<char>(token));
        }
    }

    return Codec::from_utf8(string);
}