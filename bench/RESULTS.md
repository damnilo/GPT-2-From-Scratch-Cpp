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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.033 | 0.033 | — | 0.97x | — | 0.041 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.289 | 0.169 | 0.055 | 5.29x | faster | 0.057 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 10.700 | 8.176 | 0.681 | 15.72x | faster | 0.711 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.128 | 0.086 | 0.050 | 2.55x | faster | 0.045 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.108 | 4.948 | 0.174 | 29.38x | faster | 0.189 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 54.109 | 33.039 | 1.138 | 47.56x | faster | 1.149 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 184.802 | 156.486 | 130.650 | 1.41x | faster | 137.598 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 181.987 | 161.460 | 267.682 | 0.68x | not faster | 155.472 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 562.288 | 465.159 | 224.607 | 2.50x | faster | 224.745 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 621.357 | 521.724 | 273.027 | 2.28x | faster | 275.600 | faster |

Generation appends 64 tokens to a context that is already 64 tokens long. Each path uses two warmups and the median of 10 repeats. Device weights are uploaded once, outside the timer. The CPU path still projects and applies softmax to every position. The device paths project and apply softmax only to the last position.

| Shape | CPU tokens/s | cuBLAS tokens/s | Custom tokens/s |
| --- | --- | --- | --- |
| width 32, layers 3, block 64, vocab 5256 | 10.06 | 978.78 | 1194.71 |
| width 128, layers 3, block 64, vocab 5256 | 4.35 | 834.57 | 741.10 |

