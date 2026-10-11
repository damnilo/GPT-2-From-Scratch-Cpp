//
// Created by HP on 9/29/2026.
//
#include <omp.h>
#include <unordered_map>
#include "../include/Tokenizer.h"

namespace {

struct PairHash {
    size_t operator()(const std::pair<int, int>& pair) const noexcept {
        return (static_cast<size_t>(static_cast<unsigned int>(pair.first)) << 32)
             ^ static_cast<size_t>(static_cast<unsigned int>(pair.second));
    }
};

struct MergeTarget {
    int rank;
    int id;
};

using MergeMap = std::unordered_map<std::pair<int, int>, MergeTarget, PairHash>;

MergeMap buildMergeMap(const std::vector<std::pair<std::pair<int, int>, int>>& merges) {
    MergeMap map;
    map.reserve(merges.size());

    for (size_t i = 0; i < merges.size(); i++) {
        map.emplace(merges[i].first, MergeTarget{static_cast<int>(i), merges[i].second});
    }

    return map;
}

std::vector<int> encodeChunk(std::vector<int> tokens, const MergeMap& merge_map) {
    while (tokens.size() >= 2) {
        int best_rank = -1;
        std::pair<int, int> best_pair;
        int best_id = 0;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            const auto it = merge_map.find({tokens[i], tokens[i + 1]});
            if (it == merge_map.end() || (best_rank >= 0 && it->second.rank >= best_rank)) {
                continue;
            }

            best_rank = it->second.rank;
            best_pair = it->first;
            best_id = it->second.id;
        }

        if (best_rank < 0) {
            break;
        }

        tokens = Codec::merge(tokens, best_pair, best_id);
    }

    return tokens;
}

}

std::vector<int> Tokenizer::encode(const std::wstring& wstring) {
    const auto chunks = split(wstring);
    const MergeMap merge_map = buildMergeMap(merges);
    std::vector<std::vector<int>> pieces(chunks.size());

    #pragma omp parallel
    {
        std::unordered_map<std::wstring, std::vector<int>> cache;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(chunks.size()); i++) {
            const std::wstring& chunk = chunks[static_cast<size_t>(i)];
            const auto cached = cache.find(chunk);

            if (cached != cache.end()) {
                pieces[static_cast<size_t>(i)] = cached->second;
                continue;
            }

            std::vector<int> tokens = encodeChunk(Codec::to_tokens(Codec::to_utf8(chunk)), merge_map);
            cache.emplace(chunk, tokens);
            pieces[static_cast<size_t>(i)] = std::move(tokens);
        }
    }

    size_t total = 0;
    for (const auto& piece : pieces) {
        total += piece.size();
    }

    std::vector<int> ret;
    ret.reserve(total);

    for (const auto& piece : pieces) {
        ret.insert(ret.end(), piece.begin(), piece.end());
    }

    return ret;
}

void Tokenizer::train(const std::vector<std::wstring>& texts, int num_merges) {
    merges.clear();
    int next_bytes = 256;

    std::vector<std::vector<int>> corpus;

    for (const auto& t : texts) {
        for (const auto& chunk : split(t)) {
            std::string utf8 = Codec::to_utf8(chunk);
            auto tokens = Codec::to_tokens(utf8);

            if (!tokens.empty()) {
                corpus.push_back(std::move(tokens));
            }
        }
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

const std::vector<std::pair<std::pair<int, int>, int>>& Tokenizer::getMerges() const{
    return merges;
}

void Tokenizer::setMerges(const std::vector<std::pair<std::pair<int, int>, int>>& merges) {
    this->merges = merges;
}