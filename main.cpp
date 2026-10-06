#include <iostream>
#include <vector>
#include <exception>
#include <filesystem>
#include "Tokenizer/include/FileIO.h"
#include "Tokenizer/include/Tokenizer.h"

int main() {
    std::cout << "1. Creating tokenizer..." << std::endl;

    Tokenizer t;
    std::vector<std::pair<std::pair<int, int>, int>> merges;
    bool loaded = false;

    const std::string merges_path = "Training Files/merges.txt";
    const std::string input_path = "Training Files/input.txt";
    const std::string tokens_path = "Training Files/tokens.txt";

    if (std::filesystem::exists(merges_path)) {
        std::cout << "Loading Merges..." << std::endl;
        merges = FileIO::load_merges(merges_path);
        loaded = true;
    }else {
        std::cout << "Training Merges..." << std::endl;
        auto string = FileIO::read(input_path);
        t.train(string, 5000);
        merges = t.getMerges();
    }
    t.setMerges(merges);

    std::wstring text = L"Two households, both alike in dignity, "
                        L"In fair Verona, where we lay our scene,";

    std::cout << "Encoding text..." << std::endl;
    auto vector = t.encode(text);

    for (auto i : vector) {
        std::cout << i << ' ';
    }

    std::cout << std::endl;

    std::wcout << L"Decoded vector: " << t.decode(vector) << std::endl;

    if (!loaded) {
        FileIO::write(tokens_path, merges);
        FileIO::save_merges(merges_path, merges);
        std::cout << "Saved merges to merges.txt and tokens to tokens.txt" << std::endl;
    }

    return 0;
}