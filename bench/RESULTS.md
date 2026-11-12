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
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.306 | 0.171 | 0.057 | 5.34x | faster | 0.051 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 10.630 | 8.073 | 0.708 | 15.01x | faster | 0.702 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.142 | 0.095 | 0.049 | 2.88x | faster | 0.043 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.526 | 3.496 | 0.169 | 32.67x | faster | 0.169 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 55.296 | 32.871 | 1.081 | 51.16x | faster | 1.114 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 175.294 | 154.367 | 131.965 | 1.33x | faster | 188.624 | not faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 182.426 | 165.148 | 141.753 | 1.29x | faster | 150.383 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 572.260 | 466.663 | 234.363 | 2.44x | faster | 233.606 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 642.502 | 522.807 | 280.380 | 2.29x | faster | 278.154 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.95 | 2046.89 | 2299.03 |
| width 128, layers 3, block 64, vocab 5256 | 4.19 | 1292.64 | 1215.90 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 3.132 | 0.505 | 0.439 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 19.975 | 1.254 | 0.847 | 2.518e-06 | 1.669e-06 |

Training step with device attention. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU. Attention weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 127.644 | 127.068 |
| width 128, layers 3, block 64, vocab 5256 | 243.331 | 241.716 |

Training step with the transformer block on the device. Embeddings stay on the CPU. Softmax, loss, and AdamW for the block run on the device. The custom column is the tiled kernel, passed in directly.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 8.271 | 9.800 |
| width 128, layers 3, block 64, vocab 5256 | 23.654 | 25.196 |

