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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.032 | 0.033 | — | 0.97x | — | 0.044 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.323 | 0.194 | 0.061 | 5.31x | faster | 0.055 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 11.172 | 8.570 | 0.727 | 15.37x | faster | 0.729 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.127 | 0.092 | 0.052 | 2.45x | faster | 0.046 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.575 | 3.620 | 0.183 | 30.52x | faster | 0.185 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 58.978 | 34.797 | 1.178 | 50.09x | faster | 1.168 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 194.956 | 174.397 | 202.948 | 0.96x | not faster | 149.424 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 194.071 | 181.596 | 147.328 | 1.32x | faster | 158.070 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 604.792 | 502.454 | 238.145 | 2.54x | faster | 372.347 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 639.927 | 576.977 | 289.164 | 2.21x | faster | 304.162 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.33 | 1686.19 | 2228.27 |
| width 128, layers 3, block 64, vocab 5256 | 3.89 | 1423.70 | 1287.57 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 3.282 | 0.668 | 0.631 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 10.031 | 0.721 | 0.697 | 2.518e-06 | 1.669e-06 |

Training step with device attention backward. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU path. Attention weights are uploaded on every forward. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 138.064 | 150.657 |
| width 128, layers 3, block 64, vocab 5256 | 275.213 | 292.475 |

