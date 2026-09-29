//
// Created by HP on 9/29/2026.
//

#ifndef GPT_2_FROM_SCRATCH_TOKENIZER_H
#define GPT_2_FROM_SCRATCH_TOKENIZER_H

#include "Codec.h"

#endif //GPT_2_FROM_SCRATCH_TOKENIZER_H

class Tokenizer {
    Codec codec;

    public:
    std::vector<int> tokenize(const std::wstring& utf_string);
}