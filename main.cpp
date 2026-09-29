#include <iostream>
#include <vector>
#include <exception>

#include "Transformer/include/FileIO.h"
#include "Transformer/include/Tokenizer.h"

int main() {
    std::cout << "1. Creating tokenizer..." << std::endl;

    Tokenizer t;

    std::cout << "2. Reading file..." << std::endl;

    std::vector<std::wstring> texts =
        FileIO::read(R"(C:\Users\HP\GPT-2 From Scratch\Training Files\romeo_and_juliet.txt)");

    std::cout << "3. Lines read: "
              << texts.size()
              << std::endl;

    std::cout << "4. Training..." << std::endl;

    t.train(texts, 100);

    std::cout << "5. Training finished." << std::endl;

    std::wstring text =
        L"Two households, both alike in dignity, "
        L"In fair Verona, where we lay our scene,";

    std::cout << "6. Encoding..." << std::endl;

    auto tokens = t.encode(text);

    std::cout << "7. Tokens: "
              << tokens.size()
              << std::endl;

    FileIO::write(R"(C:\Users\HP\GPT-2 From Scratch\Training Files\tokens.txt)", tokens);

    std::cout << "8. Decoding..." << std::endl;

    auto res = t.decode(tokens);

    std::wcout << res << std::endl;

    std::cout << "9. Finished." << std::endl;

    return 0;
}