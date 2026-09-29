//
// Created by HP on 9/29/2026.
//
#include "../include/FileIO.h"

#include <vector>
#include <fstream>

void FileIO::write(const std::string &filename, const std::vector<int>& content) {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("File could not be opened for writing");
    }

    for (int token : content) {
        file << token << ' ';
    }
}

std::vector<std::wstring> FileIO::read(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::wstring> content;

    if (!file) {
        throw std::runtime_error("File could not be opened for reading");
    }

    std::string line;

    while (std::getline(file, line)) {
        content.push_back(Codec::from_utf8(line));
    }

    return content;
}