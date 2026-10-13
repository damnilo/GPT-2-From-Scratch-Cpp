# GPT-2 From Scratch

A C++ implementation of a small GPT-style language model, built without PyTorch or TensorFlow. The tokenizer, tensor library, transformer blocks, training loop, checkpoints, and text generation are in place.

The trained model is intentionally small: embedding width 32, 4 heads, 3 blocks, and a context window of 64 tokens. On the Shakespeare corpus in `Training Files/input.txt` it continues the opening of the book and keeps the shape of a play (speaker names, `sir`, `hath`). It does not write a coherent new scene. Wider blocks are the change that would move that limit. Sampling tweaks will not.

## Model

`Model` runs this stack. The input is a rank-2 token tensor `[batch, sequence]`.

```text
Token IDs
  ↓
Token embedding + learned position embedding
  ↓
Transformer block × 3
  ↓
LayerNorm
  ↓
Linear to vocabulary size
  ↓
Softmax over the last axis
```

Each block is pre-norm with a residual around both sublayers:

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The feed-forward sublayer is `Linear → GeLU → Linear`, with the hidden width set to 4× the embedding width. GeLU is the tanh approximation. Attention is causal self-attention: scores for future positions are set to negative infinity before the softmax. `embedding_dim` must be divisible by `n_heads`.

`Dropout` and `ReLU` exist as layers and are not part of this stack.

`CrossEntropyLoss` compares the softmax output with a one-hot target of the same shape. The reported loss is a sum over the window, not a mean. At initialization on a window of 64 tokens and a vocabulary of about 5256, that sum is about `64 * ln(5256)`, roughly 550. `AdamW` is called as:

```cpp
optimizer.step(model.parameters(), model.gradients());
```

## Training

`main.cpp` reads `Training Files/input.txt`. It loads `Training Files/merges.txt` when that file exists, and otherwise learns 5000 byte-level BPE merges and saves them. Vocabulary size is `256 + merges.size()`.

If `Training Files/model.bin` exists, training is skipped and the checkpoint is loaded. Delete that file to train again. The checkpoint stores the model config and parameter tensors. It does not store AdamW moments, so a later run cannot resume the optimizer.

The training configuration in `main.cpp` is:

| Setting | Value |
| --- | --- |
| Embedding width | 32 |
| Heads | 4 |
| Blocks | 3 |
| Context | 64 |
| Batch | 1 |
| Epochs | 10 |
| AdamW | learning rate `0.001`, weight decay `0.01` |

One epoch walks the encoded corpus in non-overlapping windows of at most 64 tokens. The target at each position is the next token. Loss is printed every 10 steps. Weights are written once, after the last epoch.

`Training Files` is resolved from the working directory when that folder exists, and otherwise from `C:/Users/HP/GPT-2 From Scratch/Training Files`. Run the program from the project root.

## Generation

Generation takes the last row of the softmax output and samples the next token:

* temperature `0.8`, applied to log-probabilities
* top-k `40`
* nucleus (top-p) `0.9` inside that shortlist
* repetition penalty `1.3` on the last 4 tokens only

A loaded checkpoint continues from 16 tokens into the book and generates 200 new tokens. A run that has just finished training generates 1000 tokens from the first 16 tokens of the book. The context window slides, and positions inside that window always restart at 0.

A penalty over the whole context window pushes common words (`and`, `it`, `to`) down and the sampler then picks rare byte-pair fragments. Keeping the penalty on the last few tokens stops immediate repeats (`it it`) without that failure.

## Checkpoints

`Checkpoint::save` / `Checkpoint::load` use a binary snapshot:

* magic `0x47505432` and version `1`
* `vocab_size`, `block_size`, `embedding_dim`, `n_heads`, `batch_size`, `sequence_len`
* each parameter tensor: rank, dimensions, then raw floats

Load fails if the magic, version, parameter count, or a tensor shape does not match a model built from the stored config.

## Tokenizer

Byte-level BPE. Token ids `0..255` are raw UTF-8 bytes. Learned merges start at `256`. Training and encoding both pre-tokenize with the same GPT-4-style regex, so a merge never crosses a pre-token boundary.

Merge files are a custom integer format, one rule per line: `left right new_id`. This is not the official GPT-2 string merge format. An older `merges.txt` written before pre-tokenization was applied during training will not match the current encoder.

Encoding applies each merge once, left to right, in file order. Repeated pre-token chunks are cached. The full `input.txt` (about 478560 tokens with the 5000-merge vocabulary) encodes in a fraction of a second.

```cpp
Tokenizer tokenizer;
auto texts = FileIO::read("Training Files/input.txt");
tokenizer.train(texts, 5000);
auto tokens = tokenizer.encode(L"Before we proceed any further, hear me speak.");
auto decoded = tokenizer.decode(tokens);
```

## Tensors and speed

`Tensor` stores contiguous row-major `float` values. It supports broadcasting, `matmul`, `batchMatmul`, reductions (`sum`, `max`, `mean`, with `keepdims`), and `exp`, `log`, `sqrt`, `pow`.

Large elementwise loops and matrix products use OpenMP. The cutoff is 4096 elements. Embedding scatter-add stays serial, because duplicate token ids update the same row.

When the build finds `nvcc`, `matmul` and `batchMatmul` whose product of dimensions is at least 65536 run through cuBLAS. Smaller products stay on the CPU. Without a CUDA compiler the same calls use the OpenMP path. Configure and build on a machine with a GPU:

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/GPT_2_From_Scratch
```

CMake prints `CUDA GEMM enabled` when the GPU path is compiled in. The first large product then prints the same line at runtime.

## Build

C++20, CMake 3.24 or newer, and OpenMP. CLion's MinGW toolchain is enough for the CPU build.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Project status

Implemented:

* [x] Byte-level BPE tokenizer, with the same pre-tokens in training and encoding
* [x] Tensor library, including broadcasting
* [x] Embedding, Linear, LayerNorm, GeLU, Softmax, and their backward passes
* [x] Causal self-attention and multi-head attention
* [x] Pre-norm residual transformer block
* [x] Learned positional embeddings
* [x] Language-model wrapper (`Model`)
* [x] Cross-entropy and AdamW
* [x] Training over the full corpus in 64-token windows
* [x] Binary checkpoints
* [x] Sampled generation (temperature, top-k, top-p, short repetition penalty)
* [x] OpenMP on large tensor kernels
* [x] Optional cuBLAS for large matrix products

Possible next steps, none of which are required for the current pipeline to run:

* A wider model (128 or 256) if the generated text should hold a scene together. The head count has to divide the width.
* Mean loss per token, overlapping windows, and a learning rate that decays near the end of training. Those use the same blocks.
* Tying the token embedding and the output projection, as GPT-2 does.
* Saving AdamW moments in the checkpoint, and writing a snapshot at the end of each epoch. Today a stopped run keeps weights only in that process.
* Dropout inside the residual branches, only if a larger model starts copying training passages.

## Goal

The project is a from-scratch walk through a GPT-style model: tokenization, the residual block, next-token training, and sampling. The current weights are a finished small run, not a general language model.
