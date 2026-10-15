//
// Created by HP on 10/15/2026.
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "NeuralNet/Optimizers/include/AdamW.h"
#include "NumCPP/include/Tensor.h"
#include "Tokenizer/include/FileIO.h"
#include "Tokenizer/include/Tokenizer.h"
#include "Transformer/include/Checkpoint.h"
#include "Transformer/include/Model.h"

namespace {

size_t sample(const float* probs, size_t vocab, float temperature, size_t top_k, float top_p, float repetition_penalty, std::span<const int> recent, std::mt19937& rng) {
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

std::vector<int> generate(const Model& model, std::vector<int> tokens, size_t count, float temperature, std::mt19937& rng) {
    const size_t block = model.getConfig().block_size;
    const size_t vocab = model.getConfig().vocab_size;

    for (size_t i = 0; i < count; i++) {
        constexpr size_t top_k = 40;
        constexpr float repetition_penalty = 1.3f;
        constexpr float top_p = 0.9f;
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

void fillWindow(const std::vector<int>& tokens, size_t start, size_t seq, size_t vocab, Tensor& input, Tensor& target) {
    for (size_t i = 0; i < seq; i++) {
        const int current = tokens[start + i];
        const int next = tokens[start + i + 1];
        if (current < 0 || next < 0 || static_cast<size_t>(current) >= vocab || static_cast<size_t>(next) >= vocab) {
            throw std::runtime_error("Token id is outside the vocabulary");
        }

        input[i] = static_cast<float>(current);
        target[i * vocab + static_cast<size_t>(next)] = 1.0f;
    }
}

float windowSum(const Model& model, const std::vector<int>& tokens, size_t start, size_t seq, size_t vocab) {
    Tensor input({1, seq}, 0.0f);
    Tensor target({1, seq, vocab}, 0.0f);
    fillWindow(tokens, start, seq, vocab, input, target);
    const Tensor output = model.forward(input);
    return CrossEntropyLoss::loss(output, target);
}

float trainWindow(Model& model, AdamW& optimizer, const std::vector<int>& tokens, size_t start, size_t seq, size_t vocab) {
    Tensor input({1, seq}, 0.0f);
    Tensor target({1, seq, vocab}, 0.0f);
    fillWindow(tokens, start, seq, vocab, input, target);

    model.zeroGrad();
    const Tensor output = model.forward(input);
    const float loss = CrossEntropyLoss::loss(output, target);
    model.backward(CrossEntropyLoss::backward(output, target));
    optimizer.step(model.parameters(), model.gradients());
    return loss;
}

double meanOverWindows(const Model& model, const std::vector<int>& tokens, size_t block_size, size_t vocab, size_t& windows) {
    double total = 0.0;
    windows = 0;

    for (size_t start = 0; start + block_size < tokens.size(); start += block_size) {
        const float sum = windowSum(model, tokens, start, block_size, vocab);
        total += static_cast<double>(sum) / static_cast<double>(block_size);
        windows++;
    }

    if (windows == 0) {
        return 0.0;
    }

    return total / static_cast<double>(windows);
}

void reportTrend(double train_mean, double validation_mean, double previous_train, double previous_validation) {
    const bool train_fell = train_mean < previous_train;
    const bool train_rose = train_mean > previous_train;
    const bool validation_fell = validation_mean < previous_validation;
    const bool validation_rose = validation_mean > previous_validation;

    if (validation_rose && train_fell) {
        std::cout << "Validation loss rose while training loss fell: overfitting." << std::endl;
        return;
    }

    if (validation_fell && train_fell) {
        std::cout << "Both losses fell: the model is still learning." << std::endl;
        return;
    }

    if (validation_rose && train_rose) {
        std::cout << "Both losses rose." << std::endl;
        return;
    }

    if (validation_fell && train_rose) {
        std::cout << "Training loss rose while validation loss fell." << std::endl;
        return;
    }

    std::cout << "Loss was unchanged on at least one split." << std::endl;
}

std::vector<std::vector<int>> fixedPrompts(const std::vector<int>& tokens) {
    constexpr size_t prompt_len = 16;
    std::vector<std::vector<int>> prompts;

    if (tokens.size() < prompt_len) {
        return prompts;
    }

    const size_t starts[] = {0, tokens.size() / 3, (tokens.size() * 2) / 3};
    for (size_t start : starts) {
        if (start + prompt_len > tokens.size()) {
            start = tokens.size() - prompt_len;
        }
        prompts.emplace_back(tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.begin() + static_cast<std::ptrdiff_t>(start + prompt_len));
    }

    return prompts;
}

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
    const std::string checkpoint_path = (training_dir / "checkpoint.bin").string();
    auto lines = FileIO::read(input_path);

    if (std::filesystem::exists(merges_path)) {
        std::cout << "Loading Merges..." << std::endl;
        merges = FileIO::load_merges(merges_path);
    } else {
        std::cout << "Training Merges..." << std::endl;
        t.train(lines, 5000);
        merges = t.getMerges();
        FileIO::save_merges(merges_path, merges);
        std::cout << "Saved merges." << std::endl;
    }
    t.setMerges(merges);

    std::wstring text;
    for (const std::wstring& line : lines) {
        text.append(line);
    }

    const std::vector<int> corpus = t.encode(text);
    const size_t vocab = 256 + merges.size();

    if (corpus.size() < 2) {
        throw std::runtime_error("Corpus is too short to split");
    }

    const size_t split = corpus.size() * 95 / 100;
    if (split < 2 || split >= corpus.size()) {
        throw std::runtime_error("Train/validation split is empty");
    }

    const std::vector<int> train(corpus.begin(), corpus.begin() + static_cast<std::ptrdiff_t>(split));
    const std::vector<int> validation(corpus.begin() + static_cast<std::ptrdiff_t>(split), corpus.end());

    constexpr std::uint64_t seed = 1;
    constexpr size_t block_size = 64;
    constexpr size_t embedding_dim = 32;
    constexpr size_t n_heads = 4;
    constexpr size_t epochs = 10;
    constexpr float learning_rate = 0.001f;
    constexpr float weight_decay = 0.01f;
    constexpr float temperature = 0.8f;
    constexpr size_t sample_tokens = 64;

    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Split index: " << split << std::endl;
    std::cout << "Train tokens: " << train.size() << std::endl;
    std::cout << "Validation tokens: " << validation.size() << std::endl;
    std::cout << "Window length: " << block_size << std::endl;
    std::cout << "Logged loss is the cross-entropy sum divided by the number of predicted tokens, in nats per token." << std::endl;
    std::cout << "Learning rate " << learning_rate
              << " still multiplies the summed gradient, so the update size matches training on the sum." << std::endl;
    std::cout << "Sample temperature: " << temperature << std::endl;
    std::cout << "Checkpoint: " << checkpoint_path << std::endl;
    std::cout << "Training Files/model.bin is left unchanged." << std::endl;

    Tensor::seed(seed);
    std::seed_seq sample_seed{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32)
    };
    std::mt19937 sample_rng(sample_seed);

    Model model(vocab, block_size, embedding_dim, n_heads, 1, block_size);
    AdamW optimizer(learning_rate, weight_decay);
    const std::vector<std::vector<int>> prompts = fixedPrompts(train);

    bool have_previous = false;
    double previous_train = 0.0;
    double previous_validation = 0.0;
    size_t step = 0;

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double train_total = 0.0;
        size_t train_windows = 0;

        for (size_t start = 0; start + block_size < train.size(); start += block_size) {
            const float sum = trainWindow(model, optimizer, train, start, block_size, vocab);
            train_total += static_cast<double>(sum) / static_cast<double>(block_size);
            train_windows++;
            step++;
        }

        const double train_mean = train_windows == 0 ? 0.0 : train_total / static_cast<double>(train_windows);
        size_t validation_windows = 0;
        const double validation_mean = meanOverWindows(model, validation, block_size, vocab, validation_windows);

        std::cout << "Epoch " << (epoch + 1) << " train " << train_mean << " nats/token";
        if (validation_windows == 0) {
            std::cout << " validation has no full window" << std::endl;
        } else {
            std::cout << " validation " << validation_mean << " nats/token" << std::endl;
            if (have_previous) {
                reportTrend(train_mean, validation_mean, previous_train, previous_validation);
            }
            previous_validation = validation_mean;
            have_previous = true;
        }
        previous_train = train_mean;

        for (size_t i = 0; i < prompts.size(); i++) {
            const std::vector<int> sample = generate(model, prompts[i], sample_tokens, temperature, sample_rng);
            std::cout << "Epoch " << (epoch + 1) << " prompt " << (i + 1) << " temperature " << temperature << std::endl;
            std::wcout << t.decode(sample) << std::endl;
        }

        Checkpoint::save(checkpoint_path, model, optimizer, seed, static_cast<std::uint64_t>(epoch + 1), static_cast<std::uint64_t>(step));
        std::cout << "Saved checkpoint after epoch " << (epoch + 1) << std::endl;
    }

    return 0;
}
