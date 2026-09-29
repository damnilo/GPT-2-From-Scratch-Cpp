//
// Created by HP on 9/29/2026.
//

#include "../include/Codec.h"

#include <map>
#include <string>
#include <vector>

std::string Codec::to_utf8(const std::wstring& wstring) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstring);
}

std::wstring Codec::from_utf8(const std::string& utf_string) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(utf_string);
}

std::vector<int> Codec::to_bytes(const std::string& utf_string) {
    std::vector<int> ret;
    ret.reserve(utf_string.size());

    for (unsigned char c : utf_string) {
        ret.push_back(c);
    }

    return ret;
}

std::map<std::pair<int, int>, int> Codec::get_stats(const std::vector<int>& bytes) {
    std::map<std::pair<int, int>, int> ret;

    for (int i = 0; i < bytes.size() - 1; i++) {
        std::pair<int, int> pair = {bytes[i], bytes[i+1]};
        ret[pair]++;
    }

    return ret;
}

std::vector<int> Codec::merge(const std::vector<int>& bytes, const std::pair<int, int>& pair, int idx) {
    std::vector<int> ret;
    ret.reserve(bytes.size());

    for (size_t i = 0; i < bytes.size(); i++) {
        if (i < bytes.size() - 1 && bytes[i] == pair.first && bytes[i+1] == pair.second) {
            ret.push_back(idx);
            i++;
        }else {
            ret.push_back(bytes[i]);
        }
    }

    return ret;
}