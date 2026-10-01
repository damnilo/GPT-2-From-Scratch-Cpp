//
// Created by HP on 9/29/2026.
//

#ifndef GPT_2_FROM_SCRATCH_FILEIO_H
#define GPT_2_FROM_SCRATCH_FILEIO_H

#include <fstream>
#include <iostream>
#include <sstream>
#include "Codec.h"
#include <vector>
#include <utility>

class FileIO {
    public:

    static void write(const std::string &filename, const std::vector<std::pair<std::pair<int, int>, int>>& merges);
    static std::vector<std::wstring> read(const std::string &filename);

    static void save_merges(const std::string &filename, std::vector<std::pair<std::pair<int, int>, int>>& merges);
    static std::vector<std::pair<std::pair<int, int>, int>> load_merges(const std::string &filename);
};

#endif //GPT_2_FROM_SCRATCH_FILEIO_H