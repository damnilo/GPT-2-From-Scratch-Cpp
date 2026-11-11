# CPU versus CUDA

Processor: Intel(R) Xeon(R) CPU @ 2.00GHz

Logical processors: 2

OpenMP threads in the measurement: 2

GPU: Tesla T4

The cuBLAS cutoff is a dimension product of 65536 (`batch * rows * common * cols`). A smaller product stays on the CPU, so the GPU column stays empty. cuBLAS is marked faster only when its median is lower than the CPU median.

The custom column is the tiled kernel. cuBLAS is turned off for that measurement, and the kernel is used for every matmul, including products below 65536. Host and device copies are included, as they are in the cuBLAS column.

Each measurement has two warmups, then the median of 10 repeats. The CPU column is OpenMP with CUDA disabled. One thread is the same CPU code with `omp_set_num_threads(1)`. Speedup is the one-thread median divided by the median of the path that actually ran.

Width 32: attention `[1, 64, 8] @ [1, 8, 64]`, feed-forward `[64, 32] @ [32, 128]`, output `[64, 32] @ [32, 5256]`.

Width 128 and 4 heads: attention `[1, 64, 32] @ [1, 32, 64]`, feed-forward `[64, 128] @ [128, 512]`, output `[64, 128] @ [128, 5256]`.

The model has 3 blocks, a context of 64, and a vocabulary of 5256.

| Measurement | Shape | Path | Threads | 1 thread ms | CPU ms | GPU ms | Speedup vs 1 thread | cuBLAS vs CPU | Custom ms | Custom vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.031 | 0.028 | — | 1.09x | — | 0.039 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.286 | 0.171 | 0.057 | 5.05x | faster | 0.051 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 10.536 | 8.056 | 0.651 | 16.19x | faster | 0.658 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.128 | 0.092 | 0.050 | 2.57x | faster | 0.045 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.038 | 3.370 | 0.173 | 29.05x | faster | 0.181 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 54.848 | 32.746 | 1.051 | 52.21x | faster | 1.099 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 171.099 | 151.336 | 128.975 | 1.33x | faster | 142.298 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 181.536 | 165.032 | 142.112 | 1.28x | faster | 147.477 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 579.301 | 485.143 | 248.150 | 2.33x | faster | 250.933 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 627.750 | 528.795 | 294.023 | 2.14x | faster | 283.393 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.97 | 1915.57 | 1935.55 |
| width 128, layers 3, block 64, vocab 5256 | 4.23 | 1640.31 | 1308.11 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 7.237 | 0.567 | 0.481 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 9.254 | 0.641 | 0.542 | 2.518e-06 | 1.669e-06 |

Training step with device attention. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU. Attention weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 126.899 | 125.887 |
| width 128, layers 3, block 64, vocab 5256 | 236.914 | 293.226 |

Training step with device attention, device feed-forward, and the device vocabulary projection. LayerNorm, the embeddings, and AdamW stay on the CPU. Attention, feed-forward, and vocabulary weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 72.500 | 73.184 |
| width 128, layers 3, block 64, vocab 5256 | 123.366 | 121.289 |

