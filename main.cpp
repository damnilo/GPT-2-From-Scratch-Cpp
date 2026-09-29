#include <iostream>
#include <vector>

#include "Transformer/include/Tokenizer.h"

int main() {
    Tokenizer t;

    std::wstring text =
        L"Two households, both alike in dignity, In fair Verona, "
        L"where we lay our scene, From ancient grudge break to new mutiny.";

    std::vector<int> tokens = t.encode(text);

    std::cout << "Number of tokens: " << tokens.size() << '\n';

    std::cout << "Tokens:\n";
    for (int token : tokens) {
        std::cout << token << ' ';
    }

    std::cout << '\n';

    std::wstring s = t.decode(tokens);

    for (auto i : s) {
        std::wcout << i;
    }

    return 0;
}
