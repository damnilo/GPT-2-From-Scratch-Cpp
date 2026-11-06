# GPT-2 From Scratch

A small GPT-style language model in C++, with no PyTorch or TensorFlow. The tokenizer, tensor library, transformer stack, training loop, checkpoints, sampling, and a CPU benchmark are included. A separate CUDA lab checks hand-written kernels. The tiled matrix product from that lab is also a selectable path in `Tensor`, next to the CPU loops and cuBLAS.

The shipped model is small on purpose: embedding width 32, 4 heads, 3 blocks, and a context of 64 tokens. `Training Files/model.bin` is a finished run on the Shakespeare text in `Training Files/input.txt`. It continues the opening of the play and keeps the shape of a scene (speaker names, `sir`, `hath`). It does not write a coherent new scene. A wider embedding is the change that would move that limit. Sampling tweaks will not.

`train.txt` is a separate short experiment on `Training Files/tiny.txt`. It does not replace that Shakespeare checkpoint.

## Build

C++20, CMake 3.24 or newer, and OpenMP. CLion's MinGW toolchain is enough for a CPU build.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

That produces four programs:

| Program | Role |
| --- | --- |
| `GPT_2_From_Scratch` | `train`, `resume`, and `generate` |
| `bench` | CPU, cuBLAS, and custom-kernel timings |
| `GPT_2_Tests` | Reference checks for tensor math and a few layers |
| `gemm_paths` | The same matmul on the CPU, through cuBLAS, and through the custom kernel |

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

Elementwise loops use one thread below 4096 elements and OpenMP above that. Embedding scatter-add stays serial, because duplicate token ids update the same row. `matmul` and `batchMatmul` use OpenMP on the CPU. When the build includes CUDA and the product `batch * rows * common * cols` is at least 65536, those two calls go through cuBLAS. Smaller products stay on the CPU, including the width-32 attention product `[1, 64, 8] @ [1, 8, 64]` (32768). `Tensor::setAllowCustomGemm(true)` with cuBLAS turned off sends every product to the tiled kernel, including ones below that cutoff. Training and `generate` leave the switch off, so they keep the cuBLAS rule.

`bench` is not part of training. From the project root:

```bash
./build/bench
```

It warms up, takes the median of 10 runs, and writes `bench/RESULTS.md`. Each row records the shape, the path that ran under the cuBLAS rule (`one thread`, `OpenMP`, or `cuBLAS`), the thread count, the one-thread time, the OpenMP time, the cuBLAS time, and the custom-kernel time. Both GPU columns include the copies between host and device. A GPU time is called faster only when its median is smaller than the 2-thread CPU median. If `nvcc` is missing, those columns stay empty.

`gemm_paths` checks values, not time. Tolerance is `1e-4` against the CPU result.

The table below is the Colab run after the 64×64 register tile: Intel Xeon at 2.00 GHz, 2 OpenMP threads, Tesla T4. Each thread owns a 4×4 result tile and the K strip is 16 wide. CPU is the 2-thread median. One-thread times and the speedup column stay in `bench/RESULTS.md`.

| Measurement | Shape | CPU ms | cuBLAS ms | Custom ms |
| --- | --- | --- | --- | --- |
| batchMatmul | `[1, 64, 8] @ [1, 8, 64]` | 0.030 | — | 0.043 |
| matmul | `[64, 32] @ [32, 128]` | 0.169 | 0.056 | 0.052 |
| matmul | `[64, 32] @ [32, 5256]` | 8.357 | 0.901 | 0.761 |
| batchMatmul | `[1, 64, 32] @ [1, 32, 64]` | 0.086 | 0.052 | 0.048 |
| matmul | `[64, 128] @ [128, 512]` | 3.395 | 0.177 | 0.198 |
| matmul | `[64, 128] @ [128, 5256]` | 32.961 | 1.171 | 1.191 |
| forward+backward | width 32, 3 layers | 156.536 | 132.989 | 139.526 |
| full step | width 32, 3 layers | 170.248 | 157.197 | 148.918 |
| forward+backward | width 128, 3 layers | 480.821 | 233.894 | 236.432 |
| full step | width 128, 3 layers | 551.249 | 283.910 | 278.922 |

On a single product the tiled kernel is ahead of cuBLAS on `[64, 32] @ [32, 128]` (0.052 ms against 0.056 ms), on `[64, 32] @ [32, 5256]` (0.761 ms against 0.901 ms), and on `[1, 64, 32] @ [1, 32, 64]` (0.048 ms against 0.052 ms). cuBLAS is ahead on `[64, 128] @ [128, 512]` (0.177 ms against 0.198 ms). The wide vocabulary projection is a tie: 1.171 ms for cuBLAS and 1.191 ms for the tiled kernel, both about 28× the 2-thread CPU (32.961 ms). The narrower output projection is about 11× the CPU at 0.761 ms. The attention product below the cutoff stays on the CPU at 0.030 ms. The tiled call takes 0.043 ms.

On a full step at width 32 the tiled path is the fastest of the three: 148.918 ms, against 157.197 ms for the cuBLAS rule and 170.248 ms on 2 CPU threads. Forward and backward at that width goes to cuBLAS, 132.989 ms against 139.526 ms. At width 128 the two GPU paths meet. Forward and backward is 233.894 ms and 236.432 ms. A full step is 278.922 ms for the tiled kernel and 283.910 ms for cuBLAS. Both are about twice the 2-thread CPU, which takes 551.249 ms.

## CUDA lab

`cuda_lab` is its own CMake project. It does not train the model. The executable is also named `bench`, but it is not the CPU benchmark above. It links the CUDA runtime and cuBLAS, and checks custom kernels against CPU loops. GEMM checks also time a device-only cuBLAS call.

```bash
cmake -S cuda_lab -B cuda_lab/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda_lab/build -j
./cuda_lab/build/bench
```

With the Visual Studio generator the program is `cuda_lab/build/Release/bench.exe`. A mismatch prints the index and both values, and the program returns 1.

| Check | What it covers |
| --- | --- |
| Vector add and scale | 10,000,000 elements. Scale uses `1/sqrt(8)` and `1/sqrt(32)`. Tolerance `1e-5`. |
| GEMM | Naive, 16×16 tiled, and cuBLAS. Shapes `2×3 @ 3×2`, `64×8 @ 8×64`, `64×32 @ 32×128`, `64×32 @ 32×5256` (tolerance `0.01`), and `7×3 @ 3×5`. The other four use `1e-3`. |
| Batch GEMM | The same five products, batch 3, with the same tolerances. |
| Mask and softmax | Causal mask and row softmax on `[1, 64, 64]`, `[2, 64, 64]`, and `[1, 7, 7]`. |
| Epilogue | Scale, causal mask, and softmax in one kernel. `(1, 64, d=8)`, `(2, 64, d=8)`, `(1, 7, d=5)`. |
| Attention forward | One head: transpose K, tiled `Q @ Kᵀ`, epilogue, tiled `scores @ V`. `(1, 64, 8)`, `(1, 64, 32)`, `(2, 64, 8)`. |
| Attention backward | Gradients of Q, K, and V for the same three heads. Tolerance `1e-5`. |

The tiled GEMM keeps one output element in a register. A block loads 16×16 tiles of A and B into shared memory, waits on `__syncthreads`, and waits again before the next tile overwrites that memory. An index outside the matrix is stored as zero. Only an in-range thread writes C.

On a Colab GPU runtime the full lab bench, including one-head attention backward, returned 0.

The sources are `vector_add.cu`, `scale.cu`, `naive_gemm.cu`, `tiled_gemm.cu`, `batch_gemm.cu`, `cublas_v2.cu`, `mask.cu`, `softmax.cu`, `epilogue.cu`, `transpose.cu`, and `attention.cu`. `bench.cu` includes them.

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
* OpenMP on large tensor kernels, cuBLAS for products at or above 65536 when `nvcc` is present, and a tiled kernel behind `Tensor::setAllowCustomGemm`
* `bench` timings on a Tesla T4, with the CPU, cuBLAS, and the tiled kernel in `bench/RESULTS.md`
* `gemm_paths`, which checks those three matmul paths against each other
* A separate `cuda_lab`: vector add, scale, naive and tiled GEMM, batch GEMM, cuBLAS checks, causal mask, softmax, a fused score epilogue, and one-head attention forward and backward. The full bench returned 0 on a Colab GPU
