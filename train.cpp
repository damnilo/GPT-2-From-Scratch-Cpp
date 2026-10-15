#include "train.h"

#include "NeuralNet/Losses/include/CrossEntropyLoss.h"
#include "NeuralNet/Optimizers/include/AdamW.h"
#include "NumCPP/include/Tensor.h"
#include "Tokenizer/include/FileIO.h"
#include "Tokenizer/include/Tokenizer.h"
#include "Transformer/include/Checkpoint.h"
#include "Transformer/include/Model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float sample_temperature = 0.8f;
constexpr size_t train_sample_tokens = 64;
constexpr size_t prompt_length = 16;

size_t sampleToken(const float* probs, size_t vocab, float temperature, size_t top_k, float top_p, float repetition_penalty, std::span<const int> recent, std::mt19937& rng) {
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
        const size_t next = sampleToken(
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

void checkToken(int token, size_t vocab) {
    if (token < 0 || static_cast<size_t>(token) >= vocab) {
        throw std::runtime_error("Token id is outside the vocabulary");
    }
}

float windowSum(const Model& model, const std::vector<int>& tokens, size_t start, size_t seq, size_t vocab) {
    Tensor input({1, seq}, 0.0f);
    Tensor target({1, seq, vocab}, 0.0f);

    for (size_t i = 0; i < seq; i++) {
        const int current = tokens[start + i];
        const int next = tokens[start + i + 1];
        checkToken(current, vocab);
        checkToken(next, vocab);
        input[i] = static_cast<float>(current);
        target[i * vocab + static_cast<size_t>(next)] = 1.0f;
    }

    const Tensor output = model.forward(input);
    return CrossEntropyLoss::loss(output, target);
}

float trainBatch(Model& model, AdamW& optimizer, const std::vector<int>& tokens, size_t first_window, size_t batch, size_t seq, size_t vocab) {
    Tensor input({batch, seq}, 0.0f);
    Tensor target({batch, seq, vocab}, 0.0f);

    for (size_t b = 0; b < batch; b++) {
        const size_t start = (first_window + b) * seq;
        for (size_t i = 0; i < seq; i++) {
            const int current = tokens[start + i];
            const int next = tokens[start + i + 1];
            checkToken(current, vocab);
            checkToken(next, vocab);
            const size_t position = b * seq + i;
            input[position] = static_cast<float>(current);
            target[position * vocab + static_cast<size_t>(next)] = 1.0f;
        }
    }

    model.zeroGrad();
    const Tensor output = model.forward(input);
    const float loss = CrossEntropyLoss::loss(output, target);
    model.backward(CrossEntropyLoss::backward(output, target));
    optimizer.step(model.parameters(), model.gradients());
    return loss;
}

size_t windowCount(size_t tokens, size_t block_size) {
    if (block_size == 0 || tokens <= block_size) {
        return 0;
    }
    return (tokens - 1) / block_size;
}

double meanOverWindows(const Model& model, const std::vector<int>& tokens, size_t block_size, size_t vocab, size_t& windows) {
    windows = windowCount(tokens.size(), block_size);
    if (windows == 0) {
        return 0.0;
    }

    double total = 0.0;
    for (size_t window = 0; window < windows; window++) {
        const float sum = windowSum(model, tokens, window * block_size, block_size, vocab);
        total += static_cast<double>(sum) / static_cast<double>(block_size);
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
    std::vector<std::vector<int>> prompts;
    if (tokens.size() < prompt_length) {
        return prompts;
    }

    const size_t starts[] = {0, tokens.size() / 3, (tokens.size() * 2) / 3};
    for (size_t start : starts) {
        if (start + prompt_length > tokens.size()) {
            start = tokens.size() - prompt_length;
        }
        prompts.emplace_back(tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.begin() + static_cast<std::ptrdiff_t>(start + prompt_length));
    }

    return prompts;
}

std::mt19937 makeRng(std::uint64_t seed) {
    std::seed_seq sequence{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32)
    };
    return std::mt19937(sequence);
}

std::string trim(const std::string& text) {
    const size_t begin = text.find_first_not_of(" \t\r");
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = text.find_last_not_of(" \t\r");
    return text.substr(begin, end - begin + 1);
}

size_t parseSize(const std::string& key, const std::string& value) {
    try {
        size_t used = 0;
        const unsigned long long parsed = std::stoull(value, &used);
        if (used != value.size()) {
            throw std::runtime_error("Invalid value for " + key);
        }
        return static_cast<size_t>(parsed);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid value for " + key);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Invalid value for " + key);
    }
}

float parseFloat(const std::string& key, const std::string& value) {
    try {
        size_t used = 0;
        const float parsed = std::stof(value, &used);
        if (used != value.size()) {
            throw std::runtime_error("Invalid value for " + key);
        }
        return parsed;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid value for " + key);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Invalid value for " + key);
    }
}

struct Prepared {
    Tokenizer tokenizer;
    std::vector<int> train;
    std::vector<int> validation;
    size_t vocab = 0;
    size_t split = 0;
};

Prepared prepare(const TrainConfig& config, bool train_merges_if_missing) {
    std::cout << "1. Creating tokenizer..." << std::endl;

    Tokenizer tokenizer;
    std::vector<std::pair<std::pair<int, int>, int>> merges;
    const auto lines = FileIO::read(config.corpus);

    if (std::filesystem::exists(config.merges)) {
        std::cout << "Loading Merges..." << std::endl;
        merges = FileIO::load_merges(config.merges);
    } else if (train_merges_if_missing) {
        std::cout << "Training Merges..." << std::endl;
        tokenizer.train(lines, 5000);
        merges = tokenizer.getMerges();
        FileIO::save_merges(config.merges, merges);
        std::cout << "Saved merges." << std::endl;
    } else {
        throw std::runtime_error("Merges file not found: " + config.merges);
    }
    tokenizer.setMerges(merges);

    std::wstring text;
    for (const std::wstring& line : lines) {
        text.append(line);
    }

    const std::vector<int> corpus = tokenizer.encode(text);
    if (corpus.size() < 2) {
        throw std::runtime_error("Corpus is too short to split");
    }

    const size_t split = corpus.size() * 95 / 100;
    if (split < 2 || split >= corpus.size()) {
        throw std::runtime_error("Train/validation split is empty");
    }

    Prepared prepared;
    prepared.tokenizer = std::move(tokenizer);
    prepared.train.assign(corpus.begin(), corpus.begin() + static_cast<std::ptrdiff_t>(split));
    prepared.validation.assign(corpus.begin() + static_cast<std::ptrdiff_t>(split), corpus.end());
    prepared.vocab = 256 + merges.size();
    prepared.split = split;
    return prepared;
}

void logSetup(const TrainConfig& config, const Prepared& prepared, std::uint64_t seed) {
    std::cout << "n_layers: " << config.n_layers << std::endl;
    std::cout << "embedding_dim: " << config.embedding_dim << std::endl;
    std::cout << "n_heads: " << config.n_heads << std::endl;
    std::cout << "block_size: " << config.block_size << std::endl;
    std::cout << "batch_size: " << config.batch_size << std::endl;
    std::cout << "epochs: " << config.epochs << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Split index: " << prepared.split << std::endl;
    std::cout << "Train tokens: " << prepared.train.size() << std::endl;
    std::cout << "Validation tokens: " << prepared.validation.size() << std::endl;
    std::cout << "Window length: " << config.block_size << std::endl;
    std::cout << "Logged loss is the cross-entropy sum divided by the number of predicted tokens, in nats per token." << std::endl;
    std::cout << "Learning rate " << config.learning_rate
              << " still multiplies the summed gradient, so the update size matches training on the sum." << std::endl;
    std::cout << "Sample temperature: " << sample_temperature << std::endl;
    std::cout << "Checkpoint: " << config.checkpoint << std::endl;
}

void requireMatch(const ModelConfig& loaded, const TrainConfig& config, size_t vocab) {
    if (loaded.vocab_size != vocab ||
        loaded.n_layers != config.n_layers ||
        loaded.embedding_dim != config.embedding_dim ||
        loaded.n_heads != config.n_heads ||
        loaded.block_size != config.block_size ||
        loaded.batch_size != config.batch_size ||
        loaded.sequence_len != config.block_size) {
        throw std::runtime_error("Checkpoint architecture does not match the config");
    }
}

void ensureParent(const std::string& path) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void runEpochs(Model& model, AdamW& optimizer, Prepared& prepared, const TrainConfig& config, size_t start_epoch, size_t step, std::uint64_t seed) {
    std::mt19937 sample_rng = makeRng(seed);
    const std::vector<std::vector<int>> prompts = fixedPrompts(prepared.train);
    bool have_previous = false;
    double previous_train = 0.0;
    double previous_validation = 0.0;

    for (size_t epoch = start_epoch; epoch < config.epochs; epoch++) {
        double train_total = 0.0;
        size_t predicted = 0;
        const size_t windows = windowCount(prepared.train.size(), config.block_size);
        const size_t steps = windows / config.batch_size;

        for (size_t index = 0; index < steps; index++) {
            const float sum = trainBatch(model, optimizer, prepared.train, index * config.batch_size, config.batch_size, config.block_size, prepared.vocab);
            train_total += static_cast<double>(sum);
            predicted += config.batch_size * config.block_size;
            step++;
        }

        const double train_mean = predicted == 0 ? 0.0 : train_total / static_cast<double>(predicted);
        size_t validation_windows = 0;
        const double validation_mean = meanOverWindows(model, prepared.validation, config.block_size, prepared.vocab, validation_windows);

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
            const std::vector<int> sample = generate(model, prompts[i], train_sample_tokens, sample_temperature, sample_rng);
            std::cout << "Epoch " << (epoch + 1) << " prompt " << (i + 1) << " temperature " << sample_temperature << std::endl;
            std::wcout << prepared.tokenizer.decode(sample) << std::endl;
        }

        ensureParent(config.checkpoint);
        Checkpoint::save(config.checkpoint, model, optimizer, seed, static_cast<std::uint64_t>(epoch + 1), static_cast<std::uint64_t>(step));
        std::cout << "Saved checkpoint after epoch " << (epoch + 1) << std::endl;
    }
}

Model makeModel(const TrainConfig& config, size_t vocab) {
    return Model(vocab, config.block_size, config.embedding_dim, config.n_heads, config.n_layers, config.batch_size, config.block_size);
}

}

TrainConfig loadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Config could not be opened: " + path);
    }

    TrainConfig config;
    bool saw_corpus = false;
    bool saw_merges = false;
    bool saw_checkpoint = false;
    std::string line;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const size_t gap = line.find_first_of(" \t");
        if (gap == std::string::npos) {
            throw std::runtime_error("Config entry needs a value: " + line);
        }

        const std::string key = line.substr(0, gap);
        const std::string value = trim(line.substr(gap + 1));
        if (value.empty()) {
            throw std::runtime_error("Config entry needs a value: " + key);
        }

        if (key == "n_layers") {
            config.n_layers = parseSize(key, value);
        } else if (key == "embedding_dim") {
            config.embedding_dim = parseSize(key, value);
        } else if (key == "n_heads") {
            config.n_heads = parseSize(key, value);
        } else if (key == "block_size") {
            config.block_size = parseSize(key, value);
        } else if (key == "batch_size") {
            config.batch_size = parseSize(key, value);
        } else if (key == "learning_rate") {
            config.learning_rate = parseFloat(key, value);
        } else if (key == "weight_decay") {
            config.weight_decay = parseFloat(key, value);
        } else if (key == "epochs") {
            config.epochs = parseSize(key, value);
        } else if (key == "seed") {
            config.seed = parseSize(key, value);
        } else if (key == "corpus") {
            config.corpus = value;
            saw_corpus = true;
        } else if (key == "merges") {
            config.merges = value;
            saw_merges = true;
        } else if (key == "checkpoint") {
            config.checkpoint = value;
            saw_checkpoint = true;
        } else {
            throw std::runtime_error("Unknown config key: " + key);
        }
    }

    if (!saw_corpus || !saw_merges || !saw_checkpoint) {
        throw std::runtime_error("Config must set corpus, merges, and checkpoint");
    }
    if (config.n_layers == 0 || config.embedding_dim == 0 || config.n_heads == 0 || config.block_size == 0 || config.batch_size == 0 || config.epochs == 0) {
        throw std::runtime_error("Config sizes must be at least 1");
    }
    if (config.embedding_dim % config.n_heads != 0) {
        throw std::runtime_error("n_heads must divide embedding_dim");
    }
    if (!(config.learning_rate > 0.0f) || config.weight_decay < 0.0f) {
        throw std::runtime_error("learning_rate must be positive and weight_decay must not be negative");
    }

    return config;
}

void runTrain(const TrainConfig& config, bool overwrite) {
    if (!overwrite && std::filesystem::exists(config.checkpoint)) {
        throw std::runtime_error("Checkpoint already exists: " + config.checkpoint + ". Pass --overwrite to replace it.");
    }

    Prepared prepared = prepare(config, true);
    logSetup(config, prepared, config.seed);
    Tensor::seed(config.seed);
    Model model = makeModel(config, prepared.vocab);
    AdamW optimizer(config.learning_rate, config.weight_decay);
    runEpochs(model, optimizer, prepared, config, 0, 0, config.seed);
}

void runResume(const TrainConfig& config) {
    if (!std::filesystem::exists(config.checkpoint)) {
        throw std::runtime_error("Checkpoint not found: " + config.checkpoint);
    }

    Prepared prepared = prepare(config, false);
    Tensor::seed(config.seed);
    AdamW optimizer(config.learning_rate, config.weight_decay);
    std::uint64_t seed = config.seed;
    std::uint64_t epoch = 0;
    std::uint64_t step = 0;
    Model model = Checkpoint::load(config.checkpoint, optimizer, seed, epoch, step);
    requireMatch(model.getConfig(), config, prepared.vocab);

    if (seed != config.seed) {
        std::cout << "Checkpoint seed " << seed << " differs from config seed " << config.seed << ". Sampling uses the checkpoint seed." << std::endl;
    }
    logSetup(config, prepared, seed);

    if (epoch >= config.epochs) {
        std::cout << "Checkpoint already finished epoch " << epoch << " of " << config.epochs << "." << std::endl;
        return;
    }

    runEpochs(model, optimizer, prepared, config, static_cast<size_t>(epoch), static_cast<size_t>(step), seed);
}

void runGenerate(const TrainConfig& config, size_t tokens) {
    if (!std::filesystem::exists(config.checkpoint)) {
        throw std::runtime_error("Checkpoint not found: " + config.checkpoint);
    }

    Prepared prepared = prepare(config, false);
    Tensor::seed(config.seed);
    Model model = Checkpoint::load(config.checkpoint);
    requireMatch(model.getConfig(), config, prepared.vocab);

    const std::vector<std::vector<int>> prompts = fixedPrompts(prepared.train);
    std::mt19937 sample_rng = makeRng(config.seed);
    std::cout << "Sample temperature: " << sample_temperature << std::endl;
    std::cout << "Tokens: " << tokens << std::endl;

    if (prompts.empty()) {
        throw std::runtime_error("Training split is shorter than the fixed prompts");
    }

    for (size_t i = 0; i < prompts.size(); i++) {
        const std::vector<int> sample = generate(model, prompts[i], tokens, sample_temperature, sample_rng);
        std::cout << "Prompt " << (i + 1) << " temperature " << sample_temperature << std::endl;
        std::wcout << prepared.tokenizer.decode(sample) << std::endl;
    }
}
