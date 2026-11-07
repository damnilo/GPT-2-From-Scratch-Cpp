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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.055 | 0.028 | — | 1.95x | — | 0.042 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.343 | 0.188 | 0.065 | 5.27x | faster | 0.072 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 15.066 | 16.785 | 0.756 | 19.92x | faster | 0.762 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.161 | 4.002 | 0.058 | 2.78x | faster | 0.057 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.687 | 8.014 | 0.182 | 31.27x | faster | 0.206 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 62.490 | 60.190 | 1.226 | 50.95x | faster | 1.333 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 174.146 | 154.854 | 127.640 | 1.36x | faster | 136.887 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 182.007 | 240.262 | 138.131 | 1.32x | faster | 147.458 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 598.528 | 474.738 | 228.241 | 2.62x | faster | 316.015 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 622.377 | 547.649 | 276.022 | 2.25x | faster | 289.680 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 9.85 | 1688.38 | 2264.72 |
| width 128, layers 3, block 64, vocab 5256 | 4.23 | 1430.18 | 1225.09 |

