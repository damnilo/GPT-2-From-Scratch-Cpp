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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.031 | 0.028 | — | 1.09x | — | 0.041 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.303 | 0.169 | 0.058 | 5.19x | faster | 0.054 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 10.869 | 8.330 | 0.674 | 16.13x | faster | 0.671 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.143 | 0.092 | 0.049 | 2.94x | faster | 0.044 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.540 | 3.400 | 0.175 | 31.67x | faster | 0.185 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 54.795 | 33.791 | 1.095 | 50.05x | faster | 1.160 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 178.305 | 157.067 | 133.863 | 1.33x | faster | 145.166 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 188.928 | 165.885 | 145.117 | 1.30x | faster | 161.798 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 579.471 | 481.301 | 230.228 | 2.52x | faster | 233.727 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 637.495 | 532.106 | 281.224 | 2.27x | faster | 280.943 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.31 | 1703.85 | 2262.60 |
| width 128, layers 3, block 64, vocab 5256 | 4.22 | 1510.08 | 1301.91 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 5.529 | 0.582 | 0.550 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 9.329 | 0.661 | 0.621 | 2.518e-06 | 1.669e-06 |

Training step with device attention. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU. Attention weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 125.308 | 127.767 |
| width 128, layers 3, block 64, vocab 5256 | 237.813 | 328.943 |

Training step with device attention and device feed-forward. LayerNorm, embedding, the vocabulary projection, and AdamW stay on the CPU. Attention and feed-forward weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 117.201 | 120.367 |
| width 128, layers 3, block 64, vocab 5256 | 186.990 | 194.116 |

