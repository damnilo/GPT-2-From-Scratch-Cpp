#include "../include/Block.h"
#include "../include/Attention.h"
#include "../include/FeedForward.h"
#include "../include/Kernels.cuh"
#include "../include/LayerNorm.h"
#include "../include/Vocab.h"

#include "../../NeuralNet/Layers/include/Linear.h"

#include <stdexcept>

namespace {

void writeGradient(Tensor& destination, const std::vector<float>& source) {
    if (destination.size() != source.size()) {
        throw std::runtime_error("gradient size mismatch");
    }
    destination = Tensor(destination.getShape(), source);
}

void freeBuffer(float*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

bool reserve(float*& dest, size_t count) {
    dest = nullptr;
    return cudaMalloc(reinterpret_cast<void**>(&dest), count * sizeof(float)) == cudaSuccess;
}

bool forwardBlock(
    Transformer& block,
    AttentionDevice& attention,
    FeedForwardDevice& feed,
    LayerNormDevice& ln1,
    LayerNormDevice& ln2,
    float* x,
    float* y,
    size_t T,
    size_t C
) {
    if (!forwardLayerNormOnDevice(block.layerNorm1(), x, y, T, ln1.state)) return false;
    if (!forwardAttentionOnDevice(attention.path, block.attention(), y, y, T, attention.state)) return false;
    cudaAdd<<<static_cast<unsigned>(T), static_cast<unsigned>(C)>>>(x, y, x, T, C);
    if (!forwardLayerNormOnDevice(block.layerNorm2(), x, y, T, ln2.state)) return false;
    if (!forwardFeedForwardOnDevice(feed.path, block.feedForward(), y, y, T, feed.state)) return false;
    cudaAdd<<<static_cast<unsigned>(T), static_cast<unsigned>(C)>>>(x, y, x, T, C);
    return cudaGetLastError() == cudaSuccess;
}

bool backwardBlock(
    Transformer& block,
    AttentionDevice& attention,
    FeedForwardDevice& feed,
    LayerNormDevice& ln1,
    LayerNormDevice& ln2,
    float* grad,
    float* dx,
    float* temp,
    size_t T,
    size_t C
) {
    FeedForwardGrad feed_grad;
    if (!backwardFeedForwardOnDevice(feed.path, feed.state, block.feedForward(), grad, temp, feed_grad)) {
        return false;
    }
    commitFeedForwardGrad(block.feedForward(), feed_grad);

    LayerNormGrad norm_grad;
    if (!backwardLayerNormOnDevice(ln2.state, block.layerNorm2(), temp, temp, norm_grad)) {
        return false;
    }
    commitLayerNormGrad(block.layerNorm2(), norm_grad);
    cudaAdd<<<static_cast<unsigned>(T), static_cast<unsigned>(C)>>>(temp, grad, temp, T, C);

    AttentionGrad attention_grad;
    if (!backwardAttentionOnDevice(attention.path, attention.state, block.attention(), attention.ws, temp, dx, attention_grad)) {
        return false;
    }
    commitAttentionGrad(block.attention(), attention_grad);

    if (!backwardLayerNormOnDevice(ln1.state, block.layerNorm1(), dx, dx, norm_grad)) {
        return false;
    }
    commitLayerNormGrad(block.layerNorm1(), norm_grad);
    cudaAdd<<<static_cast<unsigned>(T), static_cast<unsigned>(C)>>>(dx, temp, dx, T, C);
    return cudaGetLastError() == cudaSuccess;
}

}

bool enableDeviceStack(Model& model, GemmDevicePath path, StackDevice& stack) {
    const ModelConfig config = model.getConfig();
    if (config.batch_size != 1 || config.embedding_dim == 0 || config.embedding_dim > 1024) {
        return false;
    }
    stack.path = path;
    if (!enableDeviceAttention(model, path, stack.attention)
        || !enableDeviceFeedForward(model, path, stack.feed)
        || !enableDeviceVocab(model, path, stack.vocab)) {
        disableDeviceStack(model, stack);
        return false;
    }

    stack.norms.resize(config.n_layers * 2 + 1);
    for (size_t i = 0; i < config.n_layers; ++i) {
        LayerNorm& first = model.block(i).layerNorm1();
        LayerNorm& second = model.block(i).layerNorm2();
        if (!uploadLayerNormWeights(first, stack.norms[i * 2].state)
            || !uploadLayerNormWeights(second, stack.norms[i * 2 + 1].state)) {
            disableDeviceStack(model, stack);
            return false;
        }
        first.useDevice(&stack.norms[i * 2]);
        second.useDevice(&stack.norms[i * 2 + 1]);
    }
    if (!uploadLayerNormWeights(model.finalNorm(), stack.norms.back().state)) {
        disableDeviceStack(model, stack);
        return false;
    }
    model.finalNorm().useDevice(&stack.norms.back());

    const size_t count = config.block_size * config.embedding_dim;
    if (!reserve(stack.x, count) || !reserve(stack.y, count)
        || !reserve(stack.grad, count) || !reserve(stack.spare, count)) {
        disableDeviceStack(model, stack);
        return false;
    }
    stack.capacity = config.block_size;
    stack.C = config.embedding_dim;
    stack.T = 0;
    model.useStack(&stack);
    return true;
}

void disableDeviceStack(Model& model, StackDevice& stack) {
    model.useStack(nullptr);
    freeBuffer(stack.x);
    freeBuffer(stack.y);
    freeBuffer(stack.grad);
    freeBuffer(stack.spare);
    stack.T = 0;
    stack.capacity = 0;
    stack.C = 0;
    for (size_t i = 0; i < model.blockCount() && i * 2 + 1 < stack.norms.size(); ++i) {
        model.block(i).layerNorm1().useDevice(nullptr);
        model.block(i).layerNorm2().useDevice(nullptr);
    }
    if (!stack.norms.empty()) {
        model.finalNorm().useDevice(nullptr);
    }
    for (LayerNormDevice& norm : stack.norms) {
        freeLayerNormState(norm.state);
    }
    stack.norms.clear();
    disableDeviceVocab(model, stack.vocab);
    disableDeviceFeedForward(model, stack.feed);
    disableDeviceAttention(model, stack.attention);
}

bool refreshDeviceStack(Model& model, StackDevice& stack) {
    if (!refreshAttentionWeights(model, stack.attention)
        || !refreshFeedForwardWeights(model, stack.feed)
        || !refreshVocabWeights(model, stack.vocab)) {
        return false;
    }
    for (size_t i = 0; i < model.blockCount(); ++i) {
        if (!uploadLayerNormWeights(model.block(i).layerNorm1(), stack.norms[i * 2].state)
            || !uploadLayerNormWeights(model.block(i).layerNorm2(), stack.norms[i * 2 + 1].state)) {
            return false;
        }
    }
    return uploadLayerNormWeights(model.finalNorm(), stack.norms.back().state);
}

bool forwardStackDevice(
    StackDevice& stack,
    Model& model,
    const float* host_embed,
    size_t T,
    std::vector<float>& host_logits
) {
    if (host_embed == nullptr || T == 0 || T > stack.capacity || stack.C == 0
        || stack.x == nullptr || stack.y == nullptr) {
        return false;
    }
    stack.T = T;
    if (cudaMemcpy(stack.x, host_embed, T * stack.C * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    for (size_t i = 0; i < model.blockCount(); ++i) {
        if (!forwardBlock(
                model.block(i),
                stack.attention[i],
                stack.feed[i],
                stack.norms[i * 2],
                stack.norms[i * 2 + 1],
                stack.x,
                stack.y,
                T,
                stack.C)) {
            return false;
        }
    }
    if (!forwardLayerNormOnDevice(model.finalNorm(), stack.x, stack.y, T, stack.norms.back().state)) {
        return false;
    }
    return forwardVocabOnDevice(stack.path, model.vocabLinear(), stack.y, T, stack.vocab.state, host_logits);
}

bool backwardStackDevice(
    StackDevice& stack,
    Model& model,
    const float* host_logit_grad,
    std::vector<float>& host_embed_grad
) {
    const size_t T = stack.T;
    const size_t C = stack.C;
    if (host_logit_grad == nullptr || T == 0 || T > stack.capacity || stack.grad == nullptr || stack.spare == nullptr) {
        return false;
    }
    VocabGrad vocab_grad;
    if (!backwardVocabDevice(stack.path, stack.vocab.state, model.vocabLinear(), host_logit_grad, vocab_grad, stack.grad)) {
        return false;
    }
    const std::vector<Tensor*> vocab_grads = model.vocabLinear().gradients();
    if (vocab_grads.size() != 2) {
        return false;
    }
    writeGradient(*vocab_grads[0], vocab_grad.W);
    writeGradient(*vocab_grads[1], vocab_grad.b);

    LayerNormGrad norm_grad;
    if (!backwardLayerNormOnDevice(stack.norms.back().state, model.finalNorm(), stack.grad, stack.spare, norm_grad)) {
        return false;
    }
    commitLayerNormGrad(model.finalNorm(), norm_grad);

    float* incoming = stack.spare;
    float* outgoing = stack.grad;
    for (size_t n = model.blockCount(); n > 0; --n) {
        const size_t i = n - 1;
        if (!backwardBlock(
                model.block(i),
                stack.attention[i],
                stack.feed[i],
                stack.norms[i * 2],
                stack.norms[i * 2 + 1],
                incoming,
                outgoing,
                stack.x,
                T,
                C)) {
            return false;
        }
        float* swap = incoming;
        incoming = outgoing;
        outgoing = swap;
    }

    host_embed_grad.resize(T * C);
    return cudaMemcpy(host_embed_grad.data(), incoming, T * C * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}
