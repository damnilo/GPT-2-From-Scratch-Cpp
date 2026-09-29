//
// Created by HP on 9/29/2026.
//

#include "../include/Codec.h"
#include <codecvt>
#include <map>
#include <string>
#include <vector>

std::string Codec::to_utf8(const std::wstring& wstring) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstring);
}

std::wstring Codec::from_utf8(const std::string& string) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(string);
}

std::vector<int> Codec::to_tokens(const std::string& string) {
    std::vector<int> ret;
    ret.reserve(string.size());

    for (unsigned char c : string) {
        ret.push_back(c);
    }

    return ret;
}

std::map<std::pair<int, int>, int> Codec::get_stats(const std::vector<int>& tokens) {
    std::map<std::pair<int, int>, int> ret;

    for (int i = 0; i+1 < tokens.size(); i++) {
        std::pair<int, int> pair = {tokens[i], tokens[i+1]};
        ret[pair]++;
    }

    return ret;
}

std::vector<int> Codec::merge(const std::vector<int>& tokens, const std::pair<int, int>& pair, int idx) {
    std::vector<int> ret;
    ret.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); i++) {
        if (i+1 < tokens.size() && tokens[i] == pair.first && tokens[i+1] == pair.second) {
            ret.push_back(idx);
            i++;
        }else {
            ret.push_back(tokens[i]);
        }
    }

    return ret;
}