#include "../include/Block.h"
#include "../include/Attention.h"
#include "../include/FeedForward.h"
#include "../include/Kernels.cuh"
#include "../include/LayerNorm.h"
#include "../include/Vocab.h"

#include "../../NeuralNet/Layers/include/Linear.h"
#include "../../NeuralNet/Optimizers/include/AdamW.h"

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
    const Model& model,
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

namespace {

struct AdamArgs {
    float lr = 0.0f;
    float wd = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float eps = 0.0f;
    size_t t = 0;
};

bool launchAdam(float* param, const float* grad, float* moment, float* velocity, size_t n, const AdamArgs& args) {
    if (param == nullptr || grad == nullptr || moment == nullptr || velocity == nullptr || n == 0) {
        return false;
    }
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    cudaAdamW<<<blocks, 256>>>(param, grad, moment, velocity, args.lr, args.wd, args.beta1, args.beta2, args.eps, args.t, n);
    return true;
}

bool adamAttention(const AttentionState& state, const AdamArgs& args) {
    const size_t heads = state.n_heads;
    const size_t d = state.d;
    const size_t width = state.C;
    if (heads == 0 || d == 0 || width != heads * d) {
        return false;
    }
    const size_t weight_n = heads * d * d;
    const size_t bias_n = heads * d;
    return launchAdam(state.Wq, state.dWq, state.mWq, state.vWq, weight_n, args)
        && launchAdam(state.bq, state.dbq, state.mbq, state.vbq, bias_n, args)
        && launchAdam(state.Wk, state.dWk, state.mWk, state.vWk, weight_n, args)
        && launchAdam(state.bk, state.dbk, state.mbk, state.vbk, bias_n, args)
        && launchAdam(state.Wv, state.dWv, state.mWv, state.vWv, weight_n, args)
        && launchAdam(state.bv, state.dbv, state.mbv, state.vbv, bias_n, args)
        && launchAdam(state.Wo_h, state.dWo_h, state.mWo_h, state.vWo_h, weight_n, args)
        && launchAdam(state.bo_h, state.dbo_h, state.mbo_h, state.vbo_h, bias_n, args)
        && launchAdam(state.Wo, state.dWo, state.mWo, state.vWo, width * width, args)
        && launchAdam(state.bo, state.dbo, state.mbo, state.vbo, width, args);
}

bool adamFeed(const FeedForwardState& state, const AdamArgs& args) {
    const size_t width = state.C;
    const size_t hidden = state.F;
    if (width == 0 || hidden != 4 * width) {
        return false;
    }
    return launchAdam(state.W1, state.dW1, state.mW1, state.vW1, width * hidden, args)
        && launchAdam(state.b1, state.db1, state.mb1, state.vb1, hidden, args)
        && launchAdam(state.W2, state.dW2, state.mW2, state.vW2, hidden * width, args)
        && launchAdam(state.b2, state.db2, state.mb2, state.vb2, width, args);
}

bool adamVocab(const VocabState& state, const AdamArgs& args) {
    if (state.C == 0 || state.V == 0) {
        return false;
    }
    return launchAdam(state.W, state.dW, state.mW, state.vW, state.C * state.V, args)
        && launchAdam(state.b, state.db, state.mb, state.vb, state.V, args);
}

bool adamNorm(const LayerNormState& state, const AdamArgs& args) {
    if (state.C == 0) {
        return false;
    }
    return launchAdam(state.gamma, state.dgamma, state.mgamma, state.vgamma, state.C, args)
        && launchAdam(state.beta, state.dbeta, state.mbeta, state.vbeta, state.C, args);
}

bool copyToHost(std::vector<float>& host, const float* device) {
    return device != nullptr
        && cudaMemcpy(host.data(), device, host.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool copyToDevice(float* device, const std::vector<float>& host) {
    return device != nullptr
        && cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
}

std::vector<float> toHostLayout(const std::vector<float>& device, size_t in_n, size_t out_n) {
    std::vector<float> host(device.size());
    for (size_t in = 0; in < in_n; ++in) {
        for (size_t out = 0; out < out_n; ++out) {
            host[out * in_n + in] = device[in * out_n + out];
        }
    }
    return host;
}

std::vector<float> toDeviceLayout(const Tensor& tensor) {
    const size_t out_n = tensor.getShape()[0];
    const size_t in_n = tensor.getShape()[1];
    std::vector<float> device(in_n * out_n);
    for (size_t in = 0; in < in_n; ++in) {
        for (size_t out = 0; out < out_n; ++out) {
            device[in * out_n + out] = tensor[out * in_n + in];
        }
    }
    return device;
}

struct AdamWalk {
    std::vector<Tensor*> params;
    const std::vector<Tensor>* host_m = nullptr;
    const std::vector<Tensor>* host_v = nullptr;
    std::vector<Tensor> moments;
    std::vector<Tensor> velocities;
    size_t index = 2;
    bool store = false;
};

bool moveMatrix(AdamWalk& walk, float* weight, float* moment, float* velocity) {
    if (walk.index >= walk.params.size() || walk.params[walk.index] == nullptr) {
        return false;
    }
    Tensor& param = *walk.params[walk.index];
    if (param.getShape().size() != 2 || weight == nullptr || moment == nullptr || velocity == nullptr) {
        return false;
    }
    const size_t out_n = param.getShape()[0];
    const size_t in_n = param.getShape()[1];
    if (walk.store) {
        std::vector<float> raw(in_n * out_n);
        if (!copyToHost(raw, weight)) {
            return false;
        }
        param = Tensor(param.getShape(), toHostLayout(raw, in_n, out_n));
        if (!copyToHost(raw, moment)) {
            return false;
        }
        walk.moments.push_back(Tensor({out_n, in_n}, toHostLayout(raw, in_n, out_n)));
        if (!copyToHost(raw, velocity)) {
            return false;
        }
        walk.velocities.push_back(Tensor({out_n, in_n}, toHostLayout(raw, in_n, out_n)));
        ++walk.index;
        return true;
    }

    const Tensor& src_m = (*walk.host_m)[walk.index];
    const Tensor& src_v = (*walk.host_v)[walk.index];
    if (src_m.getShape() != param.getShape() || src_v.getShape() != param.getShape()) {
        return false;
    }
    ++walk.index;
    return copyToDevice(moment, toDeviceLayout(src_m)) && copyToDevice(velocity, toDeviceLayout(src_v));
}

bool movePlain(AdamWalk& walk, float* weight, float* moment, float* velocity, size_t n) {
    if (walk.index >= walk.params.size() || walk.params[walk.index] == nullptr || n == 0) {
        return false;
    }
    Tensor& param = *walk.params[walk.index];
    if (param.size() != n || weight == nullptr || moment == nullptr || velocity == nullptr) {
        return false;
    }
    if (walk.store) {
        std::vector<float> raw(n);
        if (!copyToHost(raw, weight)) {
            return false;
        }
        param = Tensor(param.getShape(), raw);
        if (!copyToHost(raw, moment)) {
            return false;
        }
        walk.moments.push_back(Tensor(param.getShape(), raw));
        if (!copyToHost(raw, velocity)) {
            return false;
        }
        walk.velocities.push_back(Tensor(param.getShape(), raw));
        ++walk.index;
        return true;
    }

    const Tensor& src_m = (*walk.host_m)[walk.index];
    const Tensor& src_v = (*walk.host_v)[walk.index];
    if (src_m.getShape() != param.getShape() || src_v.getShape() != param.getShape()) {
        return false;
    }
    ++walk.index;
    return copyToDevice(moment, src_m.getData()) && copyToDevice(velocity, src_v.getData());
}

bool moveNorm(AdamWalk& walk, const LayerNormState& state) {
    return movePlain(walk, state.gamma, state.mgamma, state.vgamma, state.C)
        && movePlain(walk, state.beta, state.mbeta, state.vbeta, state.C);
}

bool moveAttention(AdamWalk& walk, const AttentionState& state) {
    const size_t heads = state.n_heads;
    const size_t d = state.d;
    const size_t width = state.C;
    if (heads == 0 || d == 0 || width != heads * d) {
        return false;
    }
    for (size_t head = 0; head < heads; ++head) {
        const size_t weight = head * d * d;
        const size_t bias = head * d;
        if (!moveMatrix(walk, state.Wq + weight, state.mWq + weight, state.vWq + weight)
            || !movePlain(walk, state.bq + bias, state.mbq + bias, state.vbq + bias, d)
            || !moveMatrix(walk, state.Wk + weight, state.mWk + weight, state.vWk + weight)
            || !movePlain(walk, state.bk + bias, state.mbk + bias, state.vbk + bias, d)
            || !moveMatrix(walk, state.Wv + weight, state.mWv + weight, state.vWv + weight)
            || !movePlain(walk, state.bv + bias, state.mbv + bias, state.vbv + bias, d)
            || !moveMatrix(walk, state.Wo_h + weight, state.mWo_h + weight, state.vWo_h + weight)
            || !movePlain(walk, state.bo_h + bias, state.mbo_h + bias, state.vbo_h + bias, d)) {
            return false;
        }
    }
    return moveMatrix(walk, state.Wo, state.mWo, state.vWo)
        && movePlain(walk, state.bo, state.mbo, state.vbo, width);
}

bool moveFeed(AdamWalk& walk, const FeedForwardState& state) {
    const size_t width = state.C;
    const size_t hidden = state.F;
    if (width == 0 || hidden != 4 * width) {
        return false;
    }
    return moveMatrix(walk, state.W1, state.mW1, state.vW1)
        && movePlain(walk, state.b1, state.mb1, state.vb1, hidden)
        && moveMatrix(walk, state.W2, state.mW2, state.vW2)
        && movePlain(walk, state.b2, state.mb2, state.vb2, width);
}

bool moveVocab(AdamWalk& walk, const VocabState& state) {
    if (state.C == 0 || state.V == 0) {
        return false;
    }
    return moveMatrix(walk, state.W, state.mW, state.vW)
        && movePlain(walk, state.b, state.mb, state.vb, state.V);
}

bool walkAdam(AdamWalk& walk, const StackDevice& stack) {
    if (stack.attention.size() != stack.feed.size() || stack.norms.size() != stack.attention.size() * 2 + 1) {
        return false;
    }
    for (size_t layer = 0; layer < stack.attention.size(); ++layer) {
        if (!moveNorm(walk, stack.norms[layer * 2].state)
            || !moveAttention(walk, stack.attention[layer].state)
            || !moveNorm(walk, stack.norms[layer * 2 + 1].state)
            || !moveFeed(walk, stack.feed[layer].state)) {
            return false;
        }
    }
    return moveNorm(walk, stack.norms.back().state) && moveVocab(walk, stack.vocab.state) && walk.index == walk.params.size();
}

}

bool adamDevice(StackDevice& stack, const AdamW& optimizer) {
    if (optimizer.getTimestep() == 0) {
        return false;
    }
    const AdamArgs args{
        optimizer.getLearningRate(),
        optimizer.getWeightDecay(),
        optimizer.getBeta1(),
        optimizer.getBeta2(),
        optimizer.getEpsilon(),
        optimizer.getTimestep()
    };
    if (stack.attention.size() != stack.feed.size() || stack.norms.size() != stack.attention.size() * 2 + 1) {
        return false;
    }
    for (size_t layer = 0; layer < stack.attention.size(); ++layer) {
        if (!adamNorm(stack.norms[layer * 2].state, args)
            || !adamAttention(stack.attention[layer].state, args)
            || !adamNorm(stack.norms[layer * 2 + 1].state, args)
            || !adamFeed(stack.feed[layer].state, args)) {
            return false;
        }
    }
    if (!adamNorm(stack.norms.back().state, args) || !adamVocab(stack.vocab.state, args)) {
        return false;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool loadAdamState(const Model& model, StackDevice& stack, const AdamW& optimizer) {
    const std::vector<Tensor>& moments = optimizer.getMoments();
    const std::vector<Tensor>& velocities = optimizer.getVelocities();
    if (moments.empty() && velocities.empty()) {
        return true;
    }
    AdamWalk walk;
    walk.params = model.parameters();
    if (moments.size() != walk.params.size() || velocities.size() != moments.size()) {
        return false;
    }
    walk.host_m = &moments;
    walk.host_v = &velocities;
    walk.store = false;
    return walkAdam(walk, stack);
}

bool storeAdamState(Model& model, StackDevice& stack, AdamW& optimizer) {
    AdamWalk walk;
    walk.params = model.parameters();
    if (walk.params.size() < 2) {
        return false;
    }
    const std::vector<Tensor>& host_m = optimizer.getMoments();
    const std::vector<Tensor>& host_v = optimizer.getVelocities();
    if (host_m.size() >= 2 && host_v.size() >= 2) {
        walk.moments.push_back(host_m[0]);
        walk.moments.push_back(host_m[1]);
        walk.velocities.push_back(host_v[0]);
        walk.velocities.push_back(host_v[1]);
    } else {
        walk.moments.emplace_back(walk.params[0]->getShape(), 0.0f);
        walk.moments.emplace_back(walk.params[1]->getShape(), 0.0f);
        walk.velocities.emplace_back(walk.params[0]->getShape(), 0.0f);
        walk.velocities.emplace_back(walk.params[1]->getShape(), 0.0f);
    }
    walk.store = true;
    if (!walkAdam(walk, stack) || walk.moments.size() != walk.params.size()) {
        return false;
    }
    optimizer.restore(optimizer.getTimestep(), std::move(walk.moments), std::move(walk.velocities));
    return true;
}
