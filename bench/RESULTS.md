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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.036 | 0.038 | — | 0.95x | — | 0.041 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.327 | 0.195 | 0.070 | 4.66x | faster | 0.056 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 12.324 | 9.425 | 0.728 | 16.94x | faster | 0.743 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.136 | 0.098 | 0.053 | 2.56x | faster | 0.046 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.957 | 3.867 | 0.178 | 33.42x | faster | 0.186 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 61.456 | 37.783 | 1.199 | 51.24x | faster | 1.238 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 202.504 | 174.808 | 147.012 | 1.38x | faster | 153.345 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 204.721 | 187.754 | 160.772 | 1.27x | faster | 167.639 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 628.040 | 523.429 | 361.387 | 1.74x | faster | 258.581 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 663.504 | 557.102 | 302.116 | 2.20x | faster | 298.504 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.22 | 1889.91 | 2258.34 |
| width 128, layers 3, block 64, vocab 5256 | 4.00 | 1497.40 | 1292.09 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 3.666 | 0.649 | 0.619 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 10.051 | 0.731 | 0.681 | 2.518e-06 | 1.669e-06 |

Training step with device attention backward. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU path. Attention weights are uploaded on every forward. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 135.135 | 258.877 |
| width 128, layers 3, block 64, vocab 5256 | 257.616 | 260.915 |

