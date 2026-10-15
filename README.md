# GPT-2 From Scratch

A small GPT-style language model in C++, with no PyTorch or TensorFlow. The tokenizer, tensor library, transformer stack, training loop, checkpoints, sampling, and a CPU benchmark are included.

The shipped model is small on purpose: embedding width 32, 4 heads, 3 blocks, and a context of 64 tokens. `Training Files/model.bin` is a finished run on the Shakespeare text in `Training Files/input.txt`. It continues the opening of the play and keeps the shape of a scene (speaker names, `sir`, `hath`). It does not write a coherent new scene. A wider embedding is the change that would move that limit. Sampling tweaks will not.

`train.txt` is a separate short experiment on `Training Files/tiny.txt`. It does not replace that Shakespeare checkpoint.

## Build

C++20, CMake 3.24 or newer, and OpenMP. CLion's MinGW toolchain is enough for a CPU build.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

That produces three programs:

| Program | Role |
| --- | --- |
| `GPT_2_From_Scratch` | `train`, `resume`, and `generate` |
| `bench` | CPU and, when CUDA is compiled in, cuBLAS timings |
| `GPT_2_Tests` | Reference checks for tensor math and a few layers |

Run the training program from the project root. Paths in the config file are relative to the current working directory.

On a machine with the CUDA toolkit:

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake prints `CUDA GEMM enabled` when that compiler is found. The first large matrix product then prints the same line at runtime. Without `nvcc`, matrix multiplication stays on the CPU.

## Train, resume, generate

```text
GPT_2_From_Scratch train --config train.txt
GPT_2_From_Scratch train --config train.txt --overwrite
GPT_2_From_Scratch resume --config train.txt
GPT_2_From_Scratch generate --config train.txt --tokens 200
```

`train` builds a new model from the config and the tokenizer, then trains. If the checkpoint path already exists, it stops. `--overwrite` replaces that file.

`resume` loads checkpoint version 2 or 3, including the AdamW moments, and continues from the saved epoch. A version 1 file has weights only, so resume refuses it. If the saved epoch count has already reached `epochs`, the command prints that and exits.

`generate` loads weights only. It does not step the optimizer. It encodes the corpus, takes the same three fixed prompts used during training, and samples `--tokens` new tokens for each one. The temperature in the log is `0.8`.

`train.txt` is one key per line. `#` starts a comment. A missing numeric key falls back to the default. `corpus`, `merges`, and `checkpoint` are required.

| Key | Default | Role |
| --- | --- | --- |
| `n_layers` | 3 | Transformer blocks. At least 1. |
| `embedding_dim` | 32 | Width of the residual stream. |
| `n_heads` | 4 | Must divide `embedding_dim`. |
| `block_size` | 64 | Context length. Every training window has this length. |
| `batch_size` | 1 | How many full windows form one AdamW step. |
| `learning_rate` | `0.001` | AdamW step size. |
| `weight_decay` | `0.01` | AdamW decay. |
| `epochs` | 10 | `train.txt` sets this to 1 so the included run stays short. |
| `seed` | 1 | Weight initialization and the sampling generator. |
| `corpus` | | Text file, read as UTF-8. |
| `merges` | | BPE merge table. Created on `train` if it is missing. |
| `checkpoint` | | Binary snapshot written after every epoch. |

The included file points at `Training Files/tiny.txt`, the existing `Training Files/merges.txt`, and `Training Files/experiment.bin`. To train on the full play, point `corpus` at `Training Files/input.txt` and choose a new checkpoint path. To sample from the finished Shakespeare weights, point `checkpoint` at `Training Files/model.bin` and keep the architecture at 3 layers, width 32, 4 heads, context 64, and batch 1. The vocabulary must match the merge file: `256 +` the number of merges.

`resume` and `generate` also check that the checkpoint architecture matches the config. A mismatch stops the run before any weights are applied to the wrong model.

## What the training log means

The corpus is encoded once. The last 5% of tokens is validation. The cut is `length * 95 / 100`, and both lengths are printed, so two runs of the same file share the same split.

Each step uses a full window of `block_size` tokens and predicts the next token at every position. A short tail that cannot fill a window is left out, so every window has the same shape. With `batch_size` greater than 1, that many consecutive windows are one step.

`CrossEntropyLoss` still returns a sum over the window. The log divides that sum by the number of predicted tokens and prints nats per token. On a fresh model and a vocabulary of about 5256, that mean is about `ln(5256)`, near 8.6. The learning rate stays `0.001` on the summed gradient, so the size of the update matches training against the sum. That choice is printed at the start of the run.

After each epoch the log prints the mean training loss and the mean validation loss. Validation only runs forward. It does not call `step` and does not change weights. From the second epoch onward, a rising validation loss while training loss falls is labeled overfitting. Both losses falling is labeled as the model still learning.

The same three 16-token prompts, taken from fixed offsets in the training split, are sampled for 64 new tokens at temperature `0.8`. The checkpoint is written at the end of every epoch.

## Generation

Sampling reads the distribution at the last position:

* temperature `0.8` on the log-probabilities
* top-k `40`
* nucleus (top-p) `0.9` inside that shortlist
* repetition penalty `1.3` on the last 4 tokens

The context window slides. Positions inside that window always restart at 0.

The penalty covers only the last few tokens. A penalty over the whole window pushes common words (`and`, `it`, `to`) down, and the sampler then picks rare byte-pair fragments. The short window stops an immediate repeat (`it it`) while leaving ordinary words available.

## Model

The input is a rank-2 tensor of token ids, shaped `[batch, sequence]`.

```text
Token IDs
  ↓
Token embedding + learned position embedding
  ↓
Transformer block × n_layers
  ↓
LayerNorm
  ↓
Linear to vocabulary size
  ↓
Softmax over the last axis
```

Each block is pre-norm, with a residual around both sublayers:

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The feed-forward sublayer is `Linear → GeLU → Linear`. The hidden width is 4× the embedding width. GeLU is the tanh approximation. Attention is causal: scores for future positions are set to negative infinity before the softmax.

`Dropout` and `ReLU` exist as layers. They are not in this stack.

AdamW is applied to the parameter and gradient lists from the model:

```cpp
optimizer.step(model.parameters(), model.gradients());
```

## Checkpoints

The file starts with magic `0x47505432` and a version.

| Version | Contents |
| --- | --- |
| 1 | Config and weights. `n_layers` is 3. No optimizer state. `Training Files/model.bin` is this version. |
| 2 | Version 1, plus AdamW `timestep`, moments `m` and `v`, completed epoch, step count, and seed. `n_layers` is still 3. |
| 3 | The same optimizer block, and `n_layers` stored in the config. New training runs write this version. |

`generate` reads versions 1, 2, and 3, and uses the weights. `resume` reads versions 2 and 3. A weights-only file cannot resume AdamW.

Loading fails if the magic or version is unknown, or if the parameter count or a tensor shape does not match the model built from the stored config.

## Tokenizer

Byte-level BPE. Token ids `0..255` are raw UTF-8 bytes. Learned merges start at `256`. Training and encoding use the same GPT-4-style pre-tokenizer, so a merge never crosses a pre-token boundary.

Merge files are a custom integer format, one rule per line: `left right new_id`. This is a different layout from the official GPT-2 string merge file. A `merges.txt` written before pre-tokenization was applied during training will not match the current encoder.

Encoding applies each merge once, left to right, in file order, and caches repeated pre-token chunks. With the 5000-merge vocabulary, `Training Files/input.txt` is about 478560 tokens and encodes in a fraction of a second.

```cpp
Tokenizer tokenizer;
auto texts = FileIO::read("Training Files/input.txt");
tokenizer.train(texts, 5000);
auto tokens = tokenizer.encode(L"Before we proceed any further, hear me speak.");
auto decoded = tokenizer.decode(tokens);
```

## Tensors and the benchmark

`Tensor` stores contiguous row-major `float` values. It supports broadcasting, `matmul`, `batchMatmul`, reductions (`sum`, `max`, `mean`, with `keepdims`), and `exp`, `log`, `sqrt`, `pow`.

Elementwise loops use one thread below 4096 elements and OpenMP above that. Embedding scatter-add stays serial, because duplicate token ids update the same row. `matmul` and `batchMatmul` use OpenMP on the CPU. When the build includes CUDA and the product `batch * rows * common * cols` is at least 65536, those two calls go through cuBLAS. Smaller products stay on the CPU, including the width-32 attention product `[1, 64, 8] @ [1, 8, 64]` (32768).

`bench` is not part of training. From the project root:

```bash
./build/bench
```

It warms up, takes the median of 10 runs, and writes `bench/RESULTS.md`. Each row records the shape, the path that actually ran (`jedna nit`, `OpenMP`, or `cuBLAS`), the thread count, the one-thread time, the OpenMP time, the cuBLAS time when that path ran, and the speedup against one thread. If `nvcc` is missing, the GPU column is left empty. cuBLAS is called faster only when its median is smaller than the CPU median.

On this machine (AMD Ryzen 5 7530U, 12 threads, no `nvcc`) every measured path was OpenMP. OpenMP against one thread ranged from 1.53× on the small attention product to 6.47× on the width-128 output projection. A full step at width 32 took 567 ms. The same step at width 128 took 1499 ms. The full table is in `bench/RESULTS.md`.

## Tests

`GPT_2_Tests` compares tensor add, subtract, multiply, divide (including broadcast), `matmul`, `batchMatmul`, and the forward and backward passes of `Embedding`, `Linear`, `LayerNorm`, `GeLU`, and `Softmax` with files under `Tests/references/data`. `Tests/references/numpy_tests.py` rewrites those files when an operation changes. It is not part of the build.

```bash
./build/GPT_2_Tests
```

## Status

In place:

* Byte-level BPE, with the same pre-tokens in training and encoding
* Tensor library, including broadcasting
* Embedding, Linear, LayerNorm, GeLU, Softmax, and their backward passes
* Causal self-attention and multi-head attention
* Pre-norm residual block, repeated `n_layers` times
* Learned positional embeddings
* Cross-entropy and AdamW
* Train / validation split, mean loss in nats per token, and a fixed seed
* Checkpoints that keep AdamW state, with version 1 still loadable for generation
* Text config and the `train`, `resume`, and `generate` commands
* Sampled generation (temperature, top-k, top-p, short repetition penalty)
* OpenMP on large tensor kernels, and cuBLAS for large matrix products when `nvcc` is present
* A separate `bench` program and the measurements in `bench/RESULTS.md`
