# GPT-2 From Scratch

A GPT-style language model in C++, with no PyTorch or TensorFlow. The repository contains a byte-level BPE tokenizer, a row-major tensor library, the transformer, training, checkpoints, sampling, and a benchmark.

The shipped model is small on purpose: width 32, 4 heads, 3 blocks, context 64. `Training Files/model.bin` is a finished run on the Shakespeare text in `Training Files/input.txt`. It continues the opening of a scene (speaker names, `sir`, `hath`). It does not write a new scene. A wider embedding is what moves that limit.

Matrix products have three implementations: OpenMP on the CPU, cuBLAS, and a tiled CUDA kernel. Training still calls the tensor library for LayerNorm, the feed-forward, the embeddings, the vocabulary projection, and AdamW. When a GPU is visible and `batch_size` is 1, each attention block saves a device forward and runs the attention backward on the device. The activation that continues through the block is the CPU forward of that same attention. The `generate` command, when a GPU is visible, keeps the weights on the device and runs one forward there.

## Programs

CMake builds seven programs. Run them from the repository root. Paths inside `train.txt` are relative to that directory.

| Program | What it does |
| --- | --- |
| `GPT_2_From_Scratch` | `train`, `resume`, and `generate` |
| `bench` | Times matmul, a training step, generation, and a hybrid step with device attention. Writes `bench/RESULTS.md`. |
| `gemm_paths` | Checks one matmul on the CPU, through cuBLAS, and through the tiled kernel. Tolerance `1e-4`. |
| `device_forward` | Checks the device forward against the last row of `Model::forward`. Tolerance `1e-3`. Skips with exit code 0 when no GPU is visible. |
| `device_backward` | Checks the device backward of one `MultiHeadAttention` (4 heads, widths 32 and 128, lengths 8 and 64) through cuBLAS and through the tiled kernel against the CPU backward. Tolerance `1e-3`. Skips with exit code 0 when no GPU is visible. |
| `train_step` | One full step with the attention backward on the device and the rest on the CPU, compared with a CPU step. Widths 32 and 128, cuBLAS and the tiled kernel. Tolerance `1e-3`. Skips with exit code 0 when no GPU is visible. |
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
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
```

CMake prints `CUDA GEMM enabled` when it finds `nvcc`. If it rejects the native GPU architecture, configure again with `-DCMAKE_CUDA_ARCHITECTURES=75` for a T4. Keep the build at one job. CUDA 13's `nvcc` is large enough that a parallel build on this runtime is killed, and the log ends with `Killed` and exit code 137. The same `cmake --build build -j1` continues a build that stopped there. Deleting `build` compiles every file again.

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
./build/train_step
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

During training, validation and the short samples after each epoch use the CPU forward. The training steps use device attention when the log says `Train path: custom device attention`. The `generate` command keeps the whole forward on the device.

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

When a GPU is visible and `batch_size` is 1, the log prints `Train path: custom device attention` before the first epoch. Each epoch enables that path for the training steps and disables it before validation and the samples. The attention product uses the tiled kernel, and its weights are uploaded on every attention forward. A run with no visible GPU, or with `batch_size` greater than 1, prints `Train path: cpu`.

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

## Two GPU paths

`Tensor::matmul` and `Tensor::batchMatmul` are what training uses. The values live on the host. A product whose size `batch * rows * common * cols` is at least 65536 is copied to the device, multiplied with cuBLAS, and copied back. Smaller products stay on the CPU. The width-32 attention product `[1, 64, 8] @ [1, 8, 64]` is 32768, so it stays on the CPU. `Tensor::setAllowCustomGemm(true)` together with cuBLAS turned off sends every product to the tiled kernel, including ones below that cutoff. Training leaves both switches at their defaults, so it keeps the cutoff.

`generate` does not use those switches. It uploads each weight once. A linear layer is stored on the host as `[out, in]` and is transposed once, on the way up, into `[in, out]`. Bias, LayerNorm, and the embedding tables are copied as stored. The forward then stays on the device. Q, K, V, and the per-head output projection of all four heads are one batched GEMM each. The `[C, C]` projection after the heads are joined, both feed-forward GEMMs, and the vocabulary GEMM stay single launches. Only the last position is projected to the vocabulary. Softmax of that row runs on the host, then the same sampler as the CPU path draws the next id.

The attention module has a device forward and a device backward, `backwardAttentionDevice`. Training uses both on the path above. The device forward keeps the activations. The backward reads that state, takes the incoming gradient from the host, and copies every weight gradient and the input gradient back. `train_step` compares the whole step with a CPU step. `device_backward` compares one attention backward. `bench` times both, and it also times this hybrid step.

The tiled kernel uses a 64×64 output tile. Each thread computes a 4×4 register tile, and the K strip loaded into shared memory is 16 wide. The launch block is `(16, 16)`. The batch launch uses block `(1, 16, 16)`, so `threadIdx.y` is the column thread and `threadIdx.z` is the row thread, with one batch index in `blockIdx.x`. When the row count and the column count are multiples of 64 and the shared dimension is a multiple of 16, the launch uses a second kernel: four floats per load and store, no edge checks, and two K strips in shared memory. The width-32 attention product has a shared dimension of 8, and the vocabulary projection has 5256 columns, so both stay on the checked kernel. The lab kernel further down is a different, slower tile: 16×16, one output element per thread.

Host tensors are row-major. cuBLAS is column-major, so a row-major `A @ B` is issued as `B` then `A` with `CUBLAS_OP_N` on both. The strided batch call uses the same operand order. The stride of each batch is the element count of one matrix, not a pointer.

## Measurements

`./build/bench` from the repository root warms up twice, takes the median of 10 repeats, and writes `bench/RESULTS.md`. The tables here are that file from a Colab GPU runtime: Intel Xeon at 2.00 GHz, 2 logical processors, 2 OpenMP threads, Tesla T4. That run predates the hybrid step and the aligned kernel, so the custom column below is the checked 64×64 kernel. The next bench run also writes a hybrid table: device attention, host for LayerNorm, the feed-forward, the embeddings, the vocabulary projection, and AdamW, at widths 32 and 128, for cuBLAS and for the tiled kernel. Attention weights are uploaded on every forward of that step.

Read the columns this way.

* **1 thread** is the CPU code with `omp_set_num_threads(1)`.
* **2 threads** is the same code with OpenMP left at the machine default.
* **cuBLAS** and **Custom**, in the product and training tables, include the copies to the device and back. Custom turns cuBLAS off and runs the tiled kernel for every product, including those under 65536.
* A blank cuBLAS cell is a product under 65536. The tensor path does not launch it.
* In this run the width-32 attention product, the width-128 vocabulary product, and the width-32 full step are slower with 2 threads than with 1. Compare a GPU time with the 1-thread column when that happens. The copies still dominate a full training step at width 32.

### Single products

Milliseconds. Lower is faster.

| Product | 1 thread | 2 threads | cuBLAS | Custom |
| --- | ---: | ---: | ---: | ---: |
| Attention, width 32, `[1, 64, 8] @ [1, 8, 64]` | 0.032 | 0.034 | — | 0.043 |
| Feed-forward, `[64, 32] @ [32, 128]` | 0.286 | 0.182 | 0.060 | 0.063 |
| Vocabulary, `[64, 32] @ [32, 5256]` | 11.359 | 9.250 | 0.779 | 0.799 |
| Attention, width 128, `[1, 64, 32] @ [1, 32, 64]` | 0.154 | 0.118 | 0.058 | 0.049 |
| Feed-forward, `[64, 128] @ [128, 512]` | 6.327 | 3.883 | 0.185 | 0.201 |
| Vocabulary, `[64, 128] @ [128, 5256]` | 64.460 | 65.340 | 2.267 | 1.264 |

On most products the two GPU kernels are within a few hundredths of a millisecond. The wide vocabulary projection at width 128 is the exception in this run: 1.264 ms through the tiled kernel and 2.267 ms through cuBLAS, against 64.460 ms on one CPU thread. The attention product at width 32 is too small for the cuBLAS rule. Its tiled call, copies included, takes 0.043 ms and does not beat the CPU at 0.032 ms on one thread or 0.034 ms on two.

### Training step

Milliseconds for 3 layers, context 64, vocabulary 5256, batch 1. The GPU columns still copy each product. Width 32 at this batch is mostly that traffic, so the GPU step is only about 1.3× faster than one CPU thread. At width 128 it is about 2.2× faster.

| Step | 1 thread | 2 threads | cuBLAS | Custom |
| --- | ---: | ---: | ---: | ---: |
| Forward and backward, width 32 | 184.815 | 159.230 | 135.613 | 142.800 |
| Full step, width 32 | 194.570 | 333.109 | 145.834 | 155.352 |
| Forward and backward, width 128 | 588.921 | 494.678 | 316.318 | 238.986 |
| Full step, width 128 | 631.463 | 539.044 | 292.558 | 289.452 |

A full step is the forward, the backward, and the AdamW update. At width 128 the cuBLAS step is 292.558 ms and the tiled step is 289.452 ms, both under half of the 1-thread time of 631.463 ms. The cuBLAS forward and backward at that width, 316.318 ms, is longer than its own full step, so that cell carries noise from the shared Colab host. At width 32 the cuBLAS step is 145.834 ms, next to its own forward and backward at 135.613 ms. The 2-thread full step at that width, 333.109 ms, is slower than the same step on one thread, so it is a property of this host run.

### Generation

Tokens per second while appending 64 tokens to a context that is already 64. Weights and the workspace are allocated once, outside the timer. Higher is faster.

| Model | CPU | cuBLAS | Custom |
| --- | ---: | ---: | ---: |
| Width 32, 3 layers, context 64, vocabulary 5256 | 9.80 | 1971.09 | 2229.06 |
| Width 128, same depth, context, and vocabulary | 4.22 | 1633.10 | 1110.37 |

The CPU column still projects and applies softmax at all 64 positions. The device columns do that only for the last position, which is the token the sampler reads. The comparison between cuBLAS and custom is on equal work. The comparison against the CPU column includes that extra CPU work.

The `generate` command uses the custom column. At width 32 that is 2229.06 tokens/s, about 0.45 ms per token. cuBLAS on the same resident forward is 1971.09. At width 128 the head width is 32 rather than 8, and cuBLAS leads, 1633.10 against 1110.37. The CPU rates, 9.80 and 4.22, are the path a build without `nvcc` actually runs.

Those device rates come from issuing Q, K, V, and the per-head output as one GEMM across the four heads. Before that, each head was its own launch, and the launch overhead dominated the 8×8 products at width 32.

### Attention backward

Milliseconds for one `MultiHeadAttention` backward: 4 heads, context 64. The CPU forward and the device forward state are built outside the timer. The device columns include the upload of the incoming gradient and the copy of every gradient back to the host. The diff columns are the largest absolute gap against the CPU backward, over the input gradient and every weight and bias gradient. A gap above `1e-3` counts as a failure.

| Width | CPU | cuBLAS | Custom | cuBLAS max diff | Custom max diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 3.603 | 0.651 | 0.615 | 8.345e-07 | 8.345e-07 |
| 128 | 22.320 | 0.830 | 0.787 | 2.518e-06 | 1.669e-06 |

The tiled kernel is slightly ahead at both widths: 0.615 ms against 0.651 ms at width 32, and 0.787 ms against 0.830 ms at width 128. Against the CPU that is about 6× at width 32 and about 28× at width 128. The largest gap, 2.518e-06, is nearly three orders of magnitude under the tolerance.

## Checks

`gemm_paths`, `device_forward`, `device_backward`, and `train_step` do not need `Tests/references/data`. `train_step` turns both tensor GEMM switches off, so the CPU step and the device step differ only in the attention backward.

`GPT_2_Tests` compares tensor add, subtract, multiply, divide (including broadcast), `matmul`, `batchMatmul`, and the forward and backward passes of `Embedding`, `Linear`, `LayerNorm`, `GeLU`, and `Softmax` with files under `Tests/references/data`. `Tests/references/numpy_tests.py` rewrites those files when an operation changes. It is not part of the build.

```bash
./build/GPT_2_Tests
```

## CUDA lab

`cuda_lab` has its own build. Its executable is also named `bench`. It is not the benchmark above.

```bash
cmake -S cuda_lab -B cuda_lab/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda_lab/build -j
./cuda_lab/build/bench
```

With the Visual Studio generator the program is `cuda_lab/build/Release/bench.exe`. A mismatch prints the index and both values and the program returns 1. On a Colab GPU runtime the full lab, including one-head attention backward, returned 0.

| Check | What it covers |
| --- | --- |
| Vector add and scale | 10,000,000 elements. Scale uses `1/sqrt(8)` and `1/sqrt(32)`. Tolerance `1e-5`. |
| GEMM | Naive, 16×16 tiled, and cuBLAS. Shapes `2×3 @ 3×2`, `64×8 @ 8×64`, `64×32 @ 32×128`, `64×32 @ 32×5256` (tolerance `0.01`), and `7×3 @ 3×5`. The other four use `1e-3`. |
| Batch GEMM | The same five products, batch 3, with the same tolerances. |
| Mask and softmax | Causal mask and row softmax on `[1, 64, 64]`, `[2, 64, 64]`, and `[1, 7, 7]`. |
| Epilogue | Scale, causal mask, and softmax in one kernel. `(1, 64, d=8)`, `(2, 64, d=8)`, `(1, 7, d=5)`. |
| Attention forward | One head: transpose K, tiled `Q @ Kᵀ`, epilogue, tiled `scores @ V`. `(1, 64, 8)`, `(1, 64, 32)`, `(2, 64, 8)`. |
| Attention backward | Gradients of Q, K, and V for the same three heads. Tolerance `1e-5`. |

The lab's tiled GEMM keeps one output element in a register. A block loads 16×16 tiles of A and B into shared memory, waits on `__syncthreads`, and waits again before the next tile overwrites that memory. An index outside the matrix is stored as zero. Only an in-range thread writes C. The fast 64×64 kernel used by `generate`, and the aligned kernel beside it, live in `NumCPP/src/CustomGemm.cu`, not in this lab.

The sources are `vector_add.cu`, `scale.cu`, `naive_gemm.cu`, `tiled_gemm.cu`, `batch_gemm.cu`, `cublas_v2.cu`, `mask.cu`, `softmax.cu`, `epilogue.cu`, `transpose.cu`, and `attention.cu`. `bench.cu` includes them.
