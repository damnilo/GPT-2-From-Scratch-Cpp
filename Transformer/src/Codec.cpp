//
// Created by HP on 9/29/2026.
//

#include "../include/Codec.h"
#include <string>
#include <vector>

std::string Codec::to_utf8(const std::wstring& utf_string) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(utf_string);
}

std::wstring Codec::from_utf8(const std::string& utf_string) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(utf_string);
}

std::vector<int> Codec::to_bytes(const std::string& utf_string) {
    std::vector<int> ret;
    ret.reserve(utf_string.size());

    for (unsigned char c : utf_string) {
        ret.push_back(static_cast<int>(c));
    }

    return ret;
}