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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.030 | 0.028 | — | 1.08x | — | 0.042 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.288 | 0.183 | 0.060 | 4.81x | faster | 0.054 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 10.805 | 8.153 | 0.724 | 14.93x | faster | 0.769 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.160 | 0.106 | 0.052 | 3.10x | faster | 0.046 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.191 | 3.497 | 0.175 | 29.67x | faster | 0.188 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 53.805 | 58.923 | 1.168 | 46.06x | faster | 1.208 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 272.038 | 152.138 | 129.777 | 2.10x | faster | 137.777 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 182.626 | 164.359 | 140.912 | 1.30x | faster | 147.566 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 577.997 | 481.440 | 228.312 | 2.53x | faster | 255.020 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 616.339 | 518.208 | 281.416 | 2.19x | faster | 284.733 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 10.04 | 1960.54 | 2257.62 |
| width 128, layers 3, block 64, vocab 5256 | 4.29 | 1579.45 | 1312.23 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 3.106 | 0.562 | 0.482 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 8.998 | 0.604 | 0.622 | 2.518e-06 | 1.669e-06 |

Training step with device attention. LayerNorm, feed-forward, embedding, the vocabulary projection, and AdamW stay on the CPU. Attention weights are refreshed after AdamW. The other products keep the cuBLAS rule.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 120.901 | 122.888 |
| width 128, layers 3, block 64, vocab 5256 | 231.855 | 232.704 |

Training step with the transformer block on the device. Embeddings stay on the CPU. Softmax, loss, and AdamW for the block run on the device. The custom column is the tiled kernel, passed in directly.

| Shape | cuBLAS ms | Custom ms |
| --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 5.755 | 7.216 |
| width 128, layers 3, block 64, vocab 5256 | 7.046 | 5.035 |

