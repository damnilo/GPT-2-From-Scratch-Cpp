# GPT-2 From Scratch

A GPT-style language model in C++, with no PyTorch or TensorFlow. The repository contains a byte-level BPE tokenizer, a row-major tensor library, the transformer, training, checkpoints, sampling, and a benchmark.

The shipped model is small on purpose: width 32, 4 heads, 3 blocks, context 64. `Training Files/model.bin` is a finished run on the Shakespeare text in `Training Files/input.txt`. It continues the opening of a scene (speaker names, `sir`, `hath`). It does not write a new scene. A wider embedding is what moves that limit.

Matrix products have three implementations: OpenMP on the CPU, cuBLAS, and a tiled CUDA kernel. When a GPU is visible and `batch_size` is 1, one training step keeps the transformer block, the vocabulary projection, softmax, the loss, and AdamW for those weights on the device. The embeddings stay on the CPU, and AdamW for the two embedding tables stays there with them. The `generate` command, when a GPU is visible, keeps the weights on the device and runs one forward there, on the tiled kernel.

Sample the shipped weights with the commands in the next two sections. The timings are at the end.

## Programs

CMake builds twelve programs. Run them from the repository root. Paths inside `train.txt` are relative to that directory.

| Program | What it does |
| --- | --- |
| `GPT_2_From_Scratch` | `train`, `resume`, and `generate` |
| `bench` | Times matmul, a tensor-library training step, generation, a hybrid step with device attention, and a step that keeps the transformer block on the device. Writes `bench/RESULTS.md`. |
| `gemm_paths` | Checks one matmul on the CPU, through cuBLAS, and through the tiled kernel. Tolerance `1e-4`. |
| `device_forward` | Checks the device forward against the last row of `Model::forward`. Tolerance `1e-3`. Skips with exit code 0 when no GPU is visible. |
| `device_backward`, `ffn_backward`, `vocab_backward` | One module each: attention, feed-forward, or the vocabulary projection. Widths 32 and 128, lengths 8 and 64, both GEMM paths. Tolerance `1e-3`. Skip with exit code 0 when no GPU is visible. |
| `ln_backward` | Checks one LayerNorm (widths 32 and 128, lengths 8 and 64) against the CPU forward and backward: the output, the input gradient, gamma, and beta. Tolerance `1e-3`. Skips with exit code 0 when no GPU is visible. |
| `train_step` | One device-block step against a CPU step. Widths 32 and 128, cuBLAS and the tiled kernel. Tolerance `1e-2`. Skips with exit code 0 when no GPU is visible. |
| `adam_step` | One device AdamW step against a CPU step. The block updates on the device. The two embedding tables update on the CPU. Widths 32 and 128, both GEMM paths. Tolerance `1e-2`. Skips with exit code 0 when no GPU is visible. |
| `ce_loss` | Device softmax and cross-entropy against the CPU loss and the logit gradient. Widths 32 and 128, lengths 8 and 64, both GEMM paths, plus one row under the `1e-8` probability floor. Tolerance `1e-3`. Skips with exit code 0 when no GPU is visible. |
| `GPT_2_Tests` | Tensor and layer checks against `Tests/references/data`. That directory is not in git. |

`cuda_lab` is a second CMake project. It checks kernels. It does not train the model.

## Run it on your machine

You need C++20, CMake 3.24 or newer, and OpenMP. A CPU build does not need the CUDA toolkit. CLion's MinGW toolchain is enough. Without `nvcc`, CMake prints `CUDA compiler not found; matrix multiplication stays on the CPU`, and `generate` prints `Generate path: cpu`.

From the repository root:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
.\build\GPT_2_From_Scratch.exe generate --config train.txt --tokens 200
```

The Visual Studio generator ignores `CMAKE_BUILD_TYPE`. Build and run the Release binary:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\GPT_2_From_Scratch.exe generate --config train.txt --tokens 200
```

On Linux or macOS the binary is `./build/GPT_2_From_Scratch`, with the same two CMake lines as the first block.

`train.txt` points at `Training Files/tiny.txt` and `Training Files/experiment.bin`. That checkpoint is not in the repository. `train` creates it. `generate` refuses to start until the file exists.

```powershell
.\build\GPT_2_From_Scratch.exe train --config train.txt
.\build\GPT_2_From_Scratch.exe generate --config train.txt --tokens 200
```

`epochs` in that file is 1, so the run is short and the samples are not the play. To sample the finished Shakespeare weights, point the config at the play and at `model.bin`, and leave the architecture at 3 layers, width 32, 4 heads, context 64, and batch 1. The vocabulary has to match `Training Files/merges.txt` (`256` plus the number of merges, 5256).

```text
corpus Training Files/input.txt
checkpoint Training Files/model.bin
```

`train` stops if the checkpoint path already exists. `--overwrite` replaces it. Do not point a training run at `model.bin` unless you mean to replace the shipped weights.

## Run it on another machine

The GPU run below is a Colab GPU runtime (Tesla T4, CUDA under `/usr/local/cuda`). A CPU runtime has no `nvcc`, so the build falls back to the CPU path.

Clone the repository, then in a notebook cell change into that directory. The example name is `/content/gpt2`. Each line that starts with `!` is a new shell, so an `export` on its own line does not survive. Put the whole build in one `%%bash` cell.

```bash
%%bash
apt-get update -qq
apt-get install -y -qq build-essential cmake ninja-build
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
```

CMake prints `CUDA GEMM enabled` when it finds `nvcc`. If it rejects the native GPU architecture, configure again with `-DCMAKE_CUDA_ARCHITECTURES=75` for a T4. Keep the build at one job. CUDA 13's `nvcc` is large enough that a parallel build on this runtime is killed, and the log ends with `Killed` and exit code 137. The same `cmake --build build -j1` continues a build that stopped there. Leave `build` in place when a new source is added, and configure again into that directory. Deleting it compiles every file again.

`Training Files/model.bin` is in the clone. `experiment.bin` is not. Sample the shipped model by pointing `train.txt` at the play, then run generate in the same style of cell so `PATH` is still set if you rebuild:

```bash
%%bash
sed -i 's|corpus Training Files/tiny.txt|corpus Training Files/input.txt|' train.txt
sed -i 's|checkpoint Training Files/experiment.bin|checkpoint Training Files/model.bin|' train.txt
export PATH=/usr/local/cuda/bin:$PATH
./build/GPT_2_From_Scratch generate --config train.txt --tokens 200
```

The log line `Generate path: custom device` means the weights were uploaded once and sampling used the tiled kernel. `Generate path: cpu` means the process saw no GPU, or the upload failed, and it used the CPU forward. Three prompts follow, each continued by 200 tokens at temperature 0.8.

The same binary runs the other commands:

```bash
./build/GPT_2_From_Scratch train --config train.txt
./build/GPT_2_From_Scratch train --config train.txt --overwrite
./build/GPT_2_From_Scratch resume --config train.txt
./build/bench
./build/gemm_paths
./build/device_forward
./build/device_backward
./build/ffn_backward
./build/vocab_backward
./build/ln_backward
./build/train_step
./build/adam_step
./build/ce_loss
```

`./build/bench` rewrites `bench/RESULTS.md`.

## Commands

| Command | Effect |
| --- | --- |
| `train --config train.txt` | Builds a new model and trains. Stops if the checkpoint file already exists. |
| `train --config train.txt --overwrite` | Replaces that file. |
| `resume --config train.txt` | Loads AdamW moments and continues from the saved epoch. |
| `generate --config train.txt --tokens 200` | Loads weights only. Samples 200 new tokens for each of three fixed prompts. |

`train` and `resume` do not take `--tokens`. `resume` and `generate` do not take `--overwrite`. `resume` reads checkpoint versions 2 and 3. A version 1 file has weights and no AdamW state, so resume refuses it. If the saved epoch has already reached `epochs`, resume prints that and exits.

During training, validation and the short samples after each epoch use the CPU forward. The training steps keep the transformer block on the device when the log says `Train path: cublas device block and vocabulary` or `Train path: custom device block and vocabulary`. The `generate` command keeps the whole forward on the device and uses the tiled kernel.

## Config

`train.txt` is one key per line. `#` starts a comment. A missing numeric key uses the default. `corpus`, `merges`, and `checkpoint` are required.

| Key | Default | Role |
| --- | --- | --- |
| `n_layers` | 3 | Transformer blocks. At least 1. |
| `embedding_dim` | 32 | Width of the residual stream. |
| `n_heads` | 4 | Must divide `embedding_dim`. |
| `block_size` | 64 | Context length. Every training window has this length. |
| `batch_size` | 1 | Full windows in one AdamW step. |
| `learning_rate` | `0.001` | AdamW step size, applied to the summed gradient. |
| `weight_decay` | `0.01` | AdamW decay. |
| `epochs` | 10 | `train.txt` sets 1 so the included run stays short. |
| `seed` | 1 | Weight initialization and the sampling generator. |
| `device_gemm` | `cublas` | `cublas` or `custom`. `custom` is used only when the head width `embedding_dim / n_heads` is a multiple of 16. Otherwise training uses cuBLAS. |
| `corpus` | | UTF-8 text file. |
| `merges` | | BPE merge table. `train` creates it when the file is missing. |
| `checkpoint` | | Binary snapshot written after every epoch. |

`resume` and `generate` also check that the checkpoint architecture matches the config. A mismatch stops the run before any weight is applied to the wrong model.

## Training

The corpus is encoded once. The last 5% of tokens is validation. The cut is `length * 95 / 100`, and both lengths are printed, so two runs of the same file share the split.

Each step takes a full window of `block_size` tokens and predicts the next token at every position. A tail that cannot fill a window is dropped. With `batch_size` greater than 1, that many consecutive windows are one step.

`CrossEntropyLoss` returns a sum over the window. The log divides that sum by the number of predicted tokens and prints nats per token. On a fresh model and a vocabulary of 5256 the mean starts near `ln(5256)`, about 8.6. The learning rate stays `0.001` on the summed gradient, so the update matches training against the sum. That choice is printed at the start.

After each epoch the log prints the mean training loss and the mean validation loss. Validation is forward only. From the second epoch on, a rising validation loss while training loss falls is labeled overfitting. Both losses falling is labeled as the model still learning.

The same three 16-token prompts, taken from fixed offsets in the training split, are then sampled for 64 new tokens at temperature 0.8. The checkpoint is written at the end of the epoch.

When a GPU is visible and `batch_size` is 1, the log prints `Train path: cublas device block and vocabulary` or `Train path: custom device block and vocabulary` before the first epoch. `device_gemm custom` is honored only when the head width is a multiple of 16. A head width of 8, the shipped width 32 with 4 heads, stays on cuBLAS, and the log says so when the config asked for `custom`. Each epoch uploads the block weights once, including LayerNorm gamma and beta, and allocates the stream buffers for that context. The first forward of a given length allocates the activation workspace, and later steps reuse it. Embeddings run on the CPU. The block, softmax, the loss, and AdamW for the block stay on the device for the epoch. AdamW for the two embedding tables stays on the CPU. At the end of the epoch the block weights and their AdamW state are copied back, and the buffers are freed before validation and the samples. A run with no visible GPU, or with `batch_size` greater than 1, prints `Train path: cpu`.

## Sampling

The distribution is read at the last position of the current window:

* temperature `0.8` on the log-probabilities
* top-k `40`
* nucleus (top-p) `0.9` inside that shortlist
* repetition penalty `1.3` on the last 4 tokens

The context is the last `block_size` tokens. Positions inside that window always restart at 0, including tokens that were already in the previous window. A cache of past keys and values would not match this model once the window is full and starts to slide. The benchmark starts from a full window of 64, so every timed step slides.

The penalty covers only those four tokens. A penalty over the whole window pushes common words (`and`, `it`, `to`) down, and the sampler then picks rare byte-pair fragments. The short window stops an immediate repeat and leaves ordinary words available.

## Model

The input is a rank-2 tensor of token ids, shaped `[batch, sequence]`.

```text
Token ids
  -> token embedding + learned position embedding
  -> transformer block, n_layers times
  -> LayerNorm
  -> linear projection to the vocabulary
  -> softmax over the last axis
```

Each block is pre-norm, with a residual around both sublayers:

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The feed-forward width is `4 * embedding_dim`. GeLU is the tanh approximation, `0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))`. LayerNorm uses `eps = 1e-5` and divides the variance by the width, not by `width - 1`.

Attention is causal. After `Q Kᵀ` the scores are divided by `sqrt(d)`, where `d = embedding_dim / n_heads` is the head width, not the residual width. A column above the diagonal becomes negative infinity, then softmax runs over the last axis. Each head has its own output projection of shape `[d, d]` before the heads are concatenated. `MultiHeadAttention` then applies one more linear layer of shape `[C, C]`.

`Dropout` and `ReLU` exist as layers. They are not in this stack.

AdamW steps the lists the model exposes:

```cpp
optimizer.step(model.parameters(), model.gradients());
```

Parameter order, which is also the order of the device upload: token embedding `[vocab, C]`, position embedding `[block, C]`, then per layer the first LayerNorm gamma and beta, then per head `Wq, bq, Wk, bk, Wv, bv, Wo, bo`, then the attention output projection and its bias, the second LayerNorm, and the two feed-forward linears. After the layers come the final LayerNorm and the vocabulary projection. GeLU and softmax have no parameters.

## Checkpoints

The file starts with magic `0x47505432` and a version.

| Version | Contents |
| --- | --- |
| 1 | Config and weights. `n_layers` is fixed at 3. No optimizer state. `Training Files/model.bin` is this version. |
| 2 | Version 1, plus AdamW timestep, moments `m` and `v`, completed epoch, step count, and seed. `n_layers` is still 3. |
| 3 | The same optimizer block, with `n_layers` stored in the config. New training runs write this version. |

`generate` reads versions 1, 2, and 3 and uses the weights. `resume` reads versions 2 and 3. Loading fails if the magic or version is unknown, or if a tensor shape does not match the model built from the stored config.

## Tokenizer

Byte-level BPE. Token ids `0..255` are raw UTF-8 bytes. Learned merges start at `256`. Training and encoding use the same GPT-4-style pre-tokenizer, so a merge never crosses a pre-token boundary.

Merge files are one integer rule per line, `left right new_id`. That is a different layout from the official GPT-2 string merge file. A `merges.txt` written before pre-tokenization was applied during training will not match the current encoder.

Encoding applies each merge once, left to right, in file order, and caches repeated pre-token chunks. With the 5000-merge vocabulary, `Training Files/input.txt` is about 478560 tokens.

```cpp
Tokenizer tokenizer;
auto texts = FileIO::read("Training Files/input.txt");
tokenizer.train(texts, 5000);
auto tokens = tokenizer.encode(L"Before we proceed any further, hear me speak.");
auto decoded = tokenizer.decode(tokens);
```

## Where the work runs

`Tensor::matmul` and `Tensor::batchMatmul` are what the CPU path uses, and what the bench measures in the forward-and-backward rows. The values live on the host. A product whose size `batch * rows * common * cols` is at least 65536 is copied to the device, multiplied with cuBLAS, and copied back. Smaller products stay on the CPU. The width-32 attention product `[1, 64, 8] @ [1, 8, 64]` is 32768, so it stays on the CPU. `Tensor::setAllowCustomGemm(true)` together with cuBLAS turned off sends every product to the tiled kernel, including ones below that cutoff. That path leaves both switches at their defaults, so it keeps the cutoff. The device block does not go through these functions. It launches GEMM on the device directly.

`generate` does not use those switches. It uploads each weight once. A linear layer is stored on the host as `[out, in]` and is transposed once, on the way up, into `[in, out]`. Bias, LayerNorm, and the embedding tables are copied as stored. The forward then stays on the device. Q, K, V, and the per-head output projection of all four heads are one batched GEMM each. The `[C, C]` projection after the heads are joined, both feed-forward GEMMs, and the vocabulary GEMM stay single launches. Only the last position is projected to the vocabulary. Softmax of that row runs on the host, then the same sampler as the CPU path draws the next id.

Attention, feed-forward, LayerNorm, and the vocabulary projection each have a device forward and a device backward. Attention includes the `[C, C]` projection after the heads. Feed-forward is two linears and the tanh GeLU between them. The GeLU backward uses the same central difference as the CPU layer. LayerNorm uses `eps = 1e-5`, divides the variance by the width, and saves the normalized values and the variance for the backward. `generate` uses a separate LayerNorm kernel that only writes the output. Those four modules share one device linear step. Host weights are stored `[out, in]` and uploaded as `[in, out]`. The vocabulary weight is `[V, C]` on the host and `[C, V]` on the device. Gamma, beta, and biases are copied as stored.

When a GPU is visible and the batch size is 1, one training step embeds on the CPU, copies that activation to the device once, and keeps the rest of the stack there. Each block is LayerNorm, attention, a residual add, LayerNorm, feed-forward, and a residual add. The final LayerNorm and the vocabulary projection stay on the device. Softmax and the cross-entropy loss run there over the vocabulary, and one loss scalar comes back. The logit gradient stays on the device and enters the vocabulary backward. AdamW for the block runs on the device, next to those weights. The embedding gradient comes back once, and AdamW for the two embedding tables runs on the CPU. The activation workspace is allocated on the first forward of a given length and reused until the epoch copies the block weights back and frees the buffers. `generate` stays on the tiled kernel and does not use this training stack.

## Measurements

`./build/bench` from the repository root warms up twice, takes the median of 10 repeats, and writes `bench/RESULTS.md`. The tables here are that file from a Colab GPU runtime: Intel Xeon at 2.00 GHz, 2 logical processors, 2 OpenMP threads, Tesla T4. The last table is the step that keeps the transformer block on the device. The custom column is the tiled kernel, including the aligned kernel where the shape allows it. In this file the attention backward stays under `1e-3`.

Read the columns this way.

* **1 thread** is the CPU code with `omp_set_num_threads(1)`.
* **2 threads** is the same code with OpenMP left at the machine default.
* **cuBLAS** and **Custom**, in the product and training tables, include the copies to the device and back. Custom turns cuBLAS off and runs the tiled kernel for every product, including those under 65536.
* A blank cuBLAS cell is a product under 65536. The tensor path does not launch it.

### Single products

Milliseconds. Lower is faster.

| Product | 1 thread | 2 threads | cuBLAS | Custom |
| --- | ---: | ---: | ---: | ---: |
| Attention, width 32, `[1, 64, 8] @ [1, 8, 64]` | 0.031 | 0.028 | — | 0.039 |
| Feed-forward, `[64, 32] @ [32, 128]` | 0.306 | 0.171 | 0.057 | 0.051 |
| Vocabulary, `[64, 32] @ [32, 5256]` | 10.630 | 8.073 | 0.708 | 0.702 |
| Attention, width 128, `[1, 64, 32] @ [1, 32, 64]` | 0.142 | 0.095 | 0.049 | 0.043 |
| Feed-forward, `[64, 128] @ [128, 512]` | 5.526 | 3.496 | 0.169 | 0.169 |
| Vocabulary, `[64, 128] @ [128, 5256]` | 55.296 | 32.871 | 1.081 | 1.114 |

The two GPU kernels stay within a few hundredths of a millisecond on every product. The vocabulary projection is the large one: 0.708 ms and 0.702 ms at width 32, against 10.630 ms on one CPU thread, and 1.081 ms and 1.114 ms at width 128, against 55.296 ms. The attention product at width 32 is too small for the cuBLAS rule. Its tiled call, copies included, takes 0.039 ms, next to 0.031 ms and 0.028 ms on the CPU.

### Training step

Milliseconds for 3 layers, context 64, vocabulary 5256, batch 1. These columns copy each tensor product. They are the tensor-library path, not the device block.

| Step | 1 thread | 2 threads | cuBLAS | Custom |
| --- | ---: | ---: | ---: | ---: |
| Forward and backward, width 32 | 175.294 | 154.367 | 131.965 | 188.624 |
| Full step, width 32 | 182.426 | 165.148 | 141.753 | 150.383 |
| Forward and backward, width 128 | 572.260 | 466.663 | 234.363 | 233.606 |
| Full step, width 128 | 642.502 | 522.807 | 280.380 | 278.154 |

A full step is the forward, the backward, and the AdamW update, through the tensor library. At width 128 the cuBLAS step is 280.380 ms and the tiled step is 278.154 ms, both under half of the 1-thread forward and backward at 572.260 ms. At width 32 the cuBLAS step is 141.753 ms, next to its own forward and backward at 131.965 ms, and the 1-thread full step is 182.426 ms, next to 175.294 ms for the forward and backward on one thread. The tiled forward and backward at width 32 is 188.624 ms, above the 2-thread CPU time of 154.367 ms. The full step on that kernel is 150.383 ms, next to the cuBLAS full step.

### Generation

Tokens per second while appending 64 tokens to a context that is already 64. Weights and the workspace are allocated once, outside the timer. Higher is faster.

| Model | CPU | cuBLAS | Custom |
| --- | ---: | ---: | ---: |
| Width 32, 3 layers, context 64, vocabulary 5256 | 9.95 | 2046.89 | 2299.03 |
| Width 128, same depth, context, and vocabulary | 4.19 | 1292.64 | 1215.90 |

The CPU column still projects and applies softmax at all 64 positions. The device columns do that only for the last position, which is the token the sampler reads. The comparison between cuBLAS and custom is on equal work. The comparison against the CPU column includes that extra CPU work.

The `generate` command uses the custom column. At width 32 that is 2299.03 tokens/s, ahead of cuBLAS at 2046.89. At width 128 the head width is 32 rather than 8, and cuBLAS leads, 1292.64 against 1215.90. The CPU rates, 9.95 and 4.19, are the path a build without `nvcc` actually runs.

Q, K, V, and the per-head output are one GEMM across the four heads.

### Attention backward

Milliseconds for one `MultiHeadAttention` backward: 4 heads, context 64. The CPU forward and the device forward state are built outside the timer. The device columns include the upload of the incoming gradient and the copy of every gradient back to the host. The diff columns are the largest absolute gap against the CPU backward, over the input gradient and every weight and bias gradient. A gap above `1e-3` counts as a failure.

| Width | CPU | cuBLAS | Custom | cuBLAS max diff | Custom max diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 3.132 | 0.505 | 0.439 | 8.345e-07 | 8.345e-07 |
| 128 | 19.975 | 1.254 | 0.847 | 2.518e-06 | 1.669e-06 |

At width 32 cuBLAS is 0.505 ms and the tiled kernel is 0.439 ms. At width 128 cuBLAS is 1.254 ms and the tiled kernel is 0.847 ms. The largest gap, 2.518e-06, is under `1e-3`. Both diff columns are the same as in the previous file. The CPU time at width 128 is 19.975 ms.

### Device training step

Milliseconds for one step. Weights are refreshed after AdamW. The first rows are device attention only, with the other layers on the CPU. The block rows are the training step from Where the work runs.

| Step | Width | cuBLAS | Custom |
| --- | ---: | ---: | ---: |
| Attention on device | 32 | 127.644 | 127.068 |
| Attention on device | 128 | 243.331 | 241.716 |
| Block on device | 32 | 8.271 | 9.800 |
| Block on device | 128 | 23.654 | 25.196 |

The attention-only rows keep LayerNorm, feed-forward, the vocabulary projection, the embeddings, and AdamW on the CPU, and they refresh the attention weights after AdamW. They are 127.644 ms and 127.068 ms at width 32, and 243.331 ms and 241.716 ms at width 128, next to 123.925 ms and 125.706 ms, and 236.653 ms and 233.793 ms, in the previous file.

The block row is the current training step. Embeddings stay on the CPU. Softmax, the loss, and AdamW for the block run on the device. At width 32, cuBLAS is 8.271 ms and the tiled kernel is 9.800 ms. At width 128 those cells are 23.654 ms and 25.196 ms. The previous table had 71.181 ms and 108.451 ms at width 32, and 115.306 ms and 116.612 ms at width 128, while softmax, the loss, and AdamW for the block were still on the CPU. The head width at width 32 is 8, so training selects cuBLAS. The bench still fills the custom column, because it passes that path in directly. The number to keep is 8.271 ms.

## Kernel

The tiled kernel uses a 64×64 output tile. Each thread computes a 4×4 register tile, and the K strip loaded into shared memory is 16 wide. The launch block is `(16, 16)`. The batch launch uses block `(1, 16, 16)`, so `threadIdx.y` is the column thread and `threadIdx.z` is the row thread, with one batch index in `blockIdx.x`. When the row count and the column count are multiples of 64 and the shared dimension is a multiple of 16, the launch uses a second kernel: four floats per load and store, no edge checks, and two K strips in shared memory. The width-32 attention product has a shared dimension of 8, and the vocabulary projection has 5256 columns, so both stay on the checked kernel. `generate` uses this kernel. It lives in `NumCPP/src/CustomGemm.cu`.

The `cuda_lab` tile is a different, slower one: 16×16, one output element per thread. A block loads 16×16 tiles of A and B into shared memory, waits on `__syncthreads`, and waits again before the next tile overwrites that memory. An index outside the matrix is stored as zero. Only an in-range thread writes C.

Host tensors are row-major. cuBLAS is column-major, so a row-major `A @ B` is issued as `B` then `A` with `CUBLAS_OP_N` on both. The strided batch call uses the same operand order. The stride of each batch is the element count of one matrix, not a pointer.

## Checks

`gemm_paths` and the device checks do not need `Tests/references/data`. The module tests and `train_step` turn both tensor GEMM switches off, so the CPU reference stays on OpenMP. What each one compares is in the program table.

`GPT_2_Tests` compares tensor add, subtract, multiply, divide (including broadcast), `matmul`, `batchMatmul`, and the forward and backward passes of `Embedding`, `Linear`, `LayerNorm`, `GeLU`, and `Softmax` with files under `Tests/references/data`. `Tests/references/numpy_tests.py` rewrites those files when an operation changes. It is not part of the build.

```bash
./build/GPT_2_Tests
```

## CUDA lab

`cuda_lab` is a second CMake project. Its executable is also named `bench`. It checks kernels and does not train the model. On a Colab T4 the full lab returned 0.

```bash
cmake -S cuda_lab -B cuda_lab/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda_lab/build -j
./cuda_lab/build/bench
```

With the Visual Studio generator the program is `cuda_lab/build/Release/bench.exe`. A mismatch prints the index and both values and the program returns 1.
