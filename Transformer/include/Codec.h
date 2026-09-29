//
// Created by HP on 9/29/2026.
//

#ifndef GPT_2_FROM_SCRATCH_CODEC_H
#define GPT_2_FROM_SCRATCH_CODEC_H

#include <string>
#include <codecvt>
#include <locale>
#include <vector>

#endif //GPT_2_FROM_SCRATCH_CODEC_H

class Codec {

    public:
    static std::string to_utf8(const std::wstring& code_point);
    static std::wstring from_utf8(const std::string& utf_string);

    static std::vector<int> to_bytes(const std::string& utf_string);
};