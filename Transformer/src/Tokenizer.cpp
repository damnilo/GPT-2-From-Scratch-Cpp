//
// Created by HP on 9/29/2026.
//
#include <omp.h>
#include "../include/Tokenizer.h"

std::vector<int> Tokenizer::encode(const std::wstring& wstring) {
    std::vector<int> ret;

    auto chunks = split(wstring);

    for (const auto& chunk : chunks) {
        std::string utf8 = Codec::to_utf8(chunk);
        auto tokens = Codec::to_tokens(utf8);

        for (const auto& [pair, count] : merges) {
            tokens = Codec::merge(tokens, pair, count);
        }

        ret.insert(ret.end(), tokens.begin(), tokens.end());
    }

    return ret;
}

void Tokenizer::train(const std::vector<std::wstring>& texts, int num_merges) {
    int next_bytes = 256;

    std::vector<std::vector<int>> corpus;

    for (const auto& t : texts) {
        std::string utf8 = Codec::to_utf8(t);
        corpus.push_back(Codec::to_tokens(utf8));
    }

    for (int i = 0; i < num_merges; ++i) {

        int num_threads = omp_get_max_threads();
        std::map<std::pair<int, int>, int> stats;
        std::vector<std::map<std::pair<int, int>, int>> local_stats(num_threads);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

            #pragma omp for
            for (int j = 0; j < static_cast<int>(corpus.size()); ++j) {
                auto map = Codec::get_stats(corpus[j]);

                for (const auto& [pair, count] : map) {
                    local_stats[thread_id][pair] += count;
                }
            }
        }

        for (const auto& thread_stats : local_stats) {
            for (const auto& [pair, count] : thread_stats) {
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

        #pragma omp parallel for
        for (int j = 0; j < static_cast<int>(corpus.size()); ++j) {
            corpus[j] = Codec::merge(corpus[j], best_pair, next_bytes);
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

std::vector<std::wstring> Tokenizer::split(const std::wstring& text) {
    std::vector<std::wstring> ret;

    auto begin = std::wsregex_iterator(text.begin(), text.end(), gpt4_split_pattern);
    auto end = std::wsregex_iterator();

    for (auto it = begin; it != end; ++it) {
        ret.push_back(it->str());
    }

    return ret;
}

std::vector<std::pair<std::pair<int, int>, int>> Tokenizer::getMerges() {
    return merges;
}