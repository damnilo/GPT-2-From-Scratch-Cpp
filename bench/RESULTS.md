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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.032 | 0.034 | — | 0.97x | — | 0.043 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.286 | 0.182 | 0.060 | 4.77x | faster | 0.063 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 11.359 | 9.250 | 0.779 | 14.58x | faster | 0.799 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.154 | 0.118 | 0.058 | 2.67x | faster | 0.049 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 6.327 | 3.883 | 0.185 | 34.11x | faster | 0.201 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 64.460 | 65.340 | 2.267 | 28.43x | faster | 1.264 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 184.815 | 159.230 | 135.613 | 1.36x | faster | 142.800 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 194.570 | 333.109 | 145.834 | 1.33x | faster | 155.352 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 588.921 | 494.678 | 316.318 | 1.86x | faster | 238.986 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 631.463 | 539.044 | 292.558 | 2.16x | faster | 289.452 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.80 | 1971.09 | 2229.06 |
| width 128, layers 3, block 64, vocab 5256 | 4.22 | 1633.10 | 1110.37 |

Attention backward is one module, 4 heads, and a context of 64. The CPU forward and the device forward state are outside the timer. The difference is the largest absolute gap against the CPU backward. A gap above 1e-3 is a failed comparison.

| Shape | CPU ms | cuBLAS ms | Custom ms | cuBLAS max abs diff | Custom max abs diff |
| --- | --- | --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 3.603 | 0.651 | 0.615 | 8.345e-07 | 8.345e-07 |
| width 128, layers 3, block 64, vocab 5256 | 22.320 | 0.830 | 0.787 | 2.518e-06 | 1.669e-06 |

