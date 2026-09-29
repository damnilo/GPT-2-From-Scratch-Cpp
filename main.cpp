#include "Transformer/include/Codec.h"
#include <iostream>
#include <string>

int main() {
    Codec codec;

    // 1. Test UTF-8 konverzije
    std::wstring text = L"banana Ćao";

    std::string utf8 = codec.to_utf8(text);

    std::cout << "UTF-8 bytes: ";
    for (unsigned char c : utf8) {
        std::cout << static_cast<int>(c) << ' ';
    }
    std::cout << '\n';

    // 2. Test byte conversion
    std::vector<int> bytes = codec.to_bytes(utf8);

    std::cout << "Vector<int>: ";
    for (int byte : bytes) {
        std::cout << byte << ' ';
    }
    std::cout << '\n';

    // 3. Test BPE pair statistics
    auto stats = codec.get_stats(bytes);

    std::cout << "\nPairs:\n";

    for (const auto& [pair, count] : stats) {
        std::cout << '('
                  << pair.first << ", "
                  << pair.second << ") -> "
                  << count << '\n';
    }

    return 0;
}