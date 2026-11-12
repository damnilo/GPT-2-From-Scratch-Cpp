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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.033 | 0.030 | — | 1.09x | — | 0.040 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.334 | 0.208 | 0.054 | 6.17x | faster | 0.049 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 11.185 | 8.682 | 0.664 | 16.83x | faster | 0.676 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.134 | 0.086 | 0.048 | 2.79x | faster | 0.044 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.351 | 3.463 | 0.167 | 32.05x | faster | 0.178 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 56.450 | 34.240 | 1.116 | 50.57x | faster | 1.153 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 183.343 | 160.585 | 134.462 | 1.36x | faster | 140.614 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 295.689 | 168.788 | 144.422 | 2.05x | faster | 150.758 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 622.142 | 496.558 | 240.498 | 2.59x | faster | 240.896 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 647.246 | 546.143 | 281.579 | 2.30x | faster | 289.740 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.72 | 1960.90 | 2269.72 |
| width 128, layers 3, block 64, vocab 5256 | 4.21 | 1469.60 | 1188.08 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 3.062 | 0.436 | 0.463 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 24.092 | 0.671 | 0.588 | 2.518e-06 | 1.669e-06 |

Training step with device attention. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU. Attention weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 123.925 | 125.706 |
| width 128, layers 3, block 64, vocab 5256 | 236.653 | 233.793 |

Training step with the transformer block on the device. The embeddings and AdamW stay on the CPU. Block weights are refreshed after AdamW. The custom column is the tiled kernel, passed in directly.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 71.181 | 108.451 |
| width 128, layers 3, block 64, vocab 5256 | 115.306 | 116.612 |

