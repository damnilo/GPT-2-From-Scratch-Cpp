#include <iostream>
#include <vector>
#include <exception>
#include <filesystem>
#include "Transformer/include/FileIO.h"
#include "Transformer/include/Tokenizer.h"

int main() {
    std::cout << "1. Creating tokenizer..." << std::endl;

    Tokenizer t;
    std::vector<std::pair<std::pair<int, int>, int>> merges;

    if (std::filesystem::exists("C:/Users/HP/GPT-2 From Scratch/Training Files/merges.txt")) {
        std::cout << "Loading Merges..." << std::endl;
        merges = FileIO::load_merges("C:/Users/HP/GPT-2 From Scratch/Training Files/merges.txt");
    }else {
        std::cout << "Training Merges..." << std::endl;
        auto string = FileIO::read("C:/Users/HP/GPT-2 From Scratch/Training Files/input.txt");
        t.train(string, 5000);
        merges = t.getMerges();
    }

    std::wstring text = L"Two households, both alike in dignity, "
                        L"In fair Verona, where we lay our scene,";

    std::cout << "Encoding text..." << std::endl;
    auto vector = t.encode(text);

    FileIO::write("C:/Users/HP/GPT-2 From Scratch/Training Files/tokens.txt", merges);
    FileIO::save_merges("C:/Users/HP/GPT-2 From Scratch/Training Files/merges.txt", merges);

    std::cout << "Saved merges to merges.txt and tokens to tokens.txt" << std::endl;

    return 0;
}