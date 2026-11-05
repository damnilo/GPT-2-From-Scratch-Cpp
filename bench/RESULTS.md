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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.065 | 0.033 | — | 1.96x | — | 0.035 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.337 | 0.191 | 0.064 | 5.29x | faster | 0.055 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 12.256 | 9.563 | 0.778 | 15.76x | faster | 0.831 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.166 | 0.115 | 0.056 | 2.94x | faster | 0.041 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 6.183 | 3.980 | 0.182 | 34.01x | faster | 0.188 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 62.514 | 37.996 | 1.238 | 50.51x | faster | 1.418 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 206.177 | 185.275 | 320.066 | 0.64x | not faster | 158.464 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 216.448 | 195.468 | 165.687 | 1.31x | faster | 183.830 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 657.497 | 573.296 | 276.860 | 2.37x | faster | 302.971 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 710.784 | 630.486 | 318.080 | 2.23x | faster | 348.572 | faster |

