#include "train.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage() {
    std::cerr << "Usage:\n"
              << "  GPT_2_From_Scratch train --config train.txt [--overwrite]\n"
              << "  GPT_2_From_Scratch resume --config train.txt\n"
              << "  GPT_2_From_Scratch generate --config train.txt --tokens 200\n";
}

std::string needValue(int argc, char** argv, int& index, const char* flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string(flag) + " needs a value");
    }
    index++;
    return argv[index];
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 1;
        }

        const std::string command = argv[1];
        std::string config_path;
        bool overwrite = false;
        bool saw_tokens = false;
        size_t tokens = 0;

        for (int i = 2; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--config") {
                config_path = needValue(argc, argv, i, "--config");
            } else if (arg == "--overwrite") {
                overwrite = true;
            } else if (arg == "--tokens") {
                const std::string text = needValue(argc, argv, i, "--tokens");
                size_t used = 0;
                const unsigned long long parsed = std::stoull(text, &used);
                if (used != text.size() || parsed == 0) {
                    throw std::runtime_error("--tokens must be a positive integer");
                }
                tokens = static_cast<size_t>(parsed);
                saw_tokens = true;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (config_path.empty()) {
            usage();
            return 1;
        }

        const TrainConfig config = loadConfig(config_path);
        if (command == "train") {
            if (saw_tokens) {
                throw std::runtime_error("train does not take --tokens");
            }
            runTrain(config, overwrite);
        } else if (command == "resume") {
            if (overwrite || saw_tokens) {
                throw std::runtime_error("resume only takes --config");
            }
            runResume(config);
        } else if (command == "generate") {
            if (overwrite || !saw_tokens) {
                usage();
                return 1;
            }
            runGenerate(config, tokens);
        } else {
            usage();
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    return 0;
}
