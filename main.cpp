#include <iostream>
#include <vector>
#include <exception>
#include <filesystem>

#include "NeuralNet/Layers/include/Embedding.h"
#include "NeuralNet/Layers/include/GeLU.h"
#include "NeuralNet/Layers/include/LayerNorm.h"
#include "NeuralNet/Layers/include/Linear.h"
#include "NeuralNet/Layers/include/ReLU.h"
#include "NeuralNet/Layers/include/Sequential.h"
#include "NeuralNet/Layers/include/Softmax.h"
#include "Tokenizer/include/FileIO.h"
#include "Tokenizer/include/Tokenizer.h"
#include "NeuralNet/Optimizers/include/AdamW.h"
#include "NeuralNet/Losses/include/CrossEntropyLoss.h"

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

    const size_t vocab = 256 + merges.size();

    if (vector.size() < 2) {
        std::cout << "Encoded text is too short to train a next-token step." << std::endl;
        return 0;
    }

    const size_t seq = vector.size() - 1;
    Sequential s;
    s.addLayer(std::make_unique<Embedding>(vocab, 5));
    s.addLayer(std::make_unique<Linear>(5, 256));
    s.addLayer(std::make_unique<ReLU>());
    s.addLayer(std::make_unique<Linear>(256, 256));
    s.addLayer(std::make_unique<GeLU>());
    s.addLayer(std::make_unique<LayerNorm>(256));
    s.addLayer(std::make_unique<Linear>(256, vocab));
    s.addLayer(std::make_unique<Softmax>(-1));

    Tensor input({1, seq}, 0.0f);
    Tensor target({1, seq, vocab}, 0.0f);

    for (size_t i = 0; i < seq; i++) {
        input[i] = static_cast<float>(vector[i]);
        target[i * vocab + static_cast<size_t>(vector[i + 1])] = 1.0f;
    }

    AdamW optim(0.001f, 0.01);

    for (size_t i = 0; i < 100; i++) {
        s.zeroGrad();
        Tensor output = s.forward(input);
        float loss = CrossEntropyLoss::loss(output, target);
        Tensor grad = CrossEntropyLoss::backward(output, target);
        s.backward(grad);
        auto parameters = s.parameters();
        auto gradients = s.gradients();

        optim.step(parameters, gradients);

        if (i % 10 == 0) std::cout << "Epoch : " << i << "| Loss : " << loss << std::endl;
    }

    return 0;
}
