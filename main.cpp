#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>
#include <filesystem>

#include "Tokenizer/include/FileIO.h"
#include "Tokenizer/include/Tokenizer.h"
#include "NeuralNet/Optimizers/include/AdamW.h"
#include "NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "Transformer/include/Checkpoint.h"
#include "Transformer/include/Model.h"

static size_t sample(const float* probs, size_t vocab, float temperature, size_t top_k, float top_p, float repetition_penalty, std::span<const int> recent, std::mt19937& rng) {
    std::vector<float> scores(vocab);
    for (size_t i = 0; i < vocab; i++) {
        scores[i] = std::log(std::max(probs[i], 1e-12f));
    }

    if (repetition_penalty > 1.0f) {
        std::vector<char> seen(vocab, 0);
        for (int token : recent) {
            if (token < 0) {
                continue;
            }
            const auto id = static_cast<size_t>(token);
            if (id >= vocab || seen[id] != 0) {
                continue;
            }
            seen[id] = 1;
            scores[id] *= repetition_penalty;
        }
    }

    const float inv_temperature = 1.0f / temperature;
    for (float& score : scores) {
        score *= inv_temperature;
    }

    std::vector<size_t> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    top_k = std::min(top_k, vocab);
    std::ranges::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k), order.end(), [&](size_t a, size_t b) {
        return scores[a] > scores[b];
    });

    const float max_score = scores[order[0]];
    std::vector<float> weights(top_k);
    float mass = 0.0f;
    for (size_t i = 0; i < top_k; i++) {
        weights[i] = std::exp(scores[order[i]] - max_score);
        mass += weights[i];
    }

    size_t keep = top_k;
    float cumulative = 0.0f;
    for (size_t i = 0; i < top_k; i++) {
        cumulative += weights[i];
        if (mass > 0.0f && cumulative / mass >= top_p) {
            keep = i + 1;
            break;
        }
    }

    std::discrete_distribution<size_t> distribution(weights.begin(), weights.begin() + static_cast<std::ptrdiff_t>(keep));
    return order[distribution(rng)];
}

static std::vector<int> generate(const Model& model, std::vector<int> tokens, size_t count) {
    const size_t block = model.getConfig().block_size;
    const size_t vocab = model.getConfig().vocab_size;

    std::mt19937 rng{std::random_device{}()};

    for (size_t i = 0; i < count; i++) {
        constexpr size_t top_k = 40;
        constexpr float repetition_penalty = 1.3f;
        constexpr float top_p = 0.9f;
        constexpr float temperature = 0.8f;
        constexpr size_t penalty_window = 4;
        const size_t length = std::min(block, tokens.size());
        const size_t start = tokens.size() - length;
        const size_t penalty_length = std::min(length, penalty_window);

        Tensor input({1, length}, 0.0f);
        for (size_t j = 0; j < length; j++) {
            input[j] = static_cast<float>(tokens[start + j]);
        }

        Tensor probs = model.forward(input);
        const size_t offset = (length - 1) * vocab;
        const size_t next = sample(
            &probs[offset],
            vocab,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            std::span<const int>(tokens.data() + (tokens.size() - penalty_length), penalty_length),
            rng
        );

        tokens.push_back(static_cast<int>(next));
    }

    return tokens;
}

int main() {
    std::cout << "1. Creating tokenizer..." << std::endl;

    Tokenizer t;
    std::vector<std::pair<std::pair<int, int>, int>> merges;

    const std::filesystem::path training_dir = std::filesystem::exists("Training Files")
        ? std::filesystem::path("Training Files")
        : std::filesystem::path("C:/Users/HP/GPT-2 From Scratch/Training Files");
    const std::string merges_path = (training_dir / "merges.txt").string();
    const std::string input_path = (training_dir / "input.txt").string();
    const std::string checkpoint_path = (training_dir / "model.bin").string();
    auto string = FileIO::read(input_path);

    if (std::filesystem::exists(merges_path)) {
        std::cout << "Loading Merges..." << std::endl;
        merges = FileIO::load_merges(merges_path);
    } else {
        std::cout << "Training Merges..." << std::endl;
        t.train(string, 5000);
        merges = t.getMerges();
        FileIO::save_merges(merges_path, merges);
        std::cout << "Saved merges." << std::endl;
    }
    t.setMerges(merges);

    std::wstring wstring;

    for (const std::wstring& s : string) {
        wstring.append(s);
    }

    auto vector = t.encode(wstring);
    const size_t vocab = 256 + merges.size();

    if (std::filesystem::exists(checkpoint_path)) {
        Model model = Checkpoint::load(checkpoint_path);

        if (model.getConfig().vocab_size != vocab) {
            throw std::runtime_error("Checkpoint vocabulary does not match the tokenizer merges");
        }

        std::cout << "Loaded model weights." << std::endl;

        const size_t prompt_len = std::min(vector.size(), static_cast<size_t>(16));
        std::vector<int> prompt(vector.begin() + 16, vector.begin() + 16 + static_cast<std::ptrdiff_t>(prompt_len));
        prompt = generate(model, prompt, 200);

        std::wcout << L"Generated text: " << t.decode(prompt) << std::endl;
        return 0;
    }

    if (vector.size() < 2) {
        std::cout << "Encoded text is too short to train a next-token step." << std::endl;
        return 0;
    }

    constexpr size_t block_size = 64;
    constexpr size_t embedding_dim = 32;
    constexpr size_t n_heads = 4;
    constexpr size_t epochs = 10;

    Model model(vocab, block_size, embedding_dim, n_heads, 1, block_size);
    AdamW optim(0.001f, 0.01f);

    std::cout << "Training on " << vector.size() << " tokens." << std::endl;

    size_t step = 0;
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        for (size_t start = 0; start + 1 < vector.size(); ) {
            const size_t seq = std::min(block_size, vector.size() - start - 1);
            Tensor input({1, seq}, 0.0f);
            Tensor target({1, seq, vocab}, 0.0f);

            for (size_t i = 0; i < seq; i++) {
                input[i] = static_cast<float>(vector[start + i]);
                target[i * vocab + static_cast<size_t>(vector[start + i + 1])] = 1.0f;
            }

            model.zeroGrad();
            Tensor output = model.forward(input);
            const float loss = CrossEntropyLoss::loss(output, target);
            model.backward(CrossEntropyLoss::backward(output, target));

            auto parameters = model.parameters();
            auto gradients = model.gradients();
            optim.step(parameters, gradients);

            if (step % 10 == 0) {
                std::cout << "Step : " << step << "| Loss : " << loss << std::endl;
            }

            start += seq;
            step++;
        }
    }

    Checkpoint::save(checkpoint_path, model);
    std::cout << "Saved model weights." << std::endl;

    const size_t prompt_len = std::min(vector.size(), static_cast<size_t>(16));
    std::vector<int> prompt(vector.begin(), vector.begin() + static_cast<std::ptrdiff_t>(prompt_len));
    prompt = generate(model, prompt, 1000);
    std::wcout << L"Generated text: " << t.decode(prompt) << std::endl;

    return 0;
}
