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
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 2 | 0.033 | 0.030 | — | 1.09x | — | 0.043 | not faster |
| matmul | [64, 32] @ [32, 128] | cuBLAS | 2 | 0.302 | 0.169 | 0.056 | 5.36x | faster | 0.052 | faster |
| matmul | [64, 32] @ [32, 5256] | cuBLAS | 2 | 10.699 | 8.357 | 0.901 | 11.87x | faster | 0.761 | faster |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | cuBLAS | 2 | 0.128 | 0.086 | 0.052 | 2.45x | faster | 0.048 | faster |
| matmul | [64, 128] @ [128, 512] | cuBLAS | 2 | 5.303 | 3.395 | 0.177 | 29.95x | faster | 0.198 | faster |
| matmul | [64, 128] @ [128, 5256] | cuBLAS | 2 | 55.048 | 32.961 | 1.171 | 47.03x | faster | 1.191 | faster |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 274.871 | 156.536 | 132.989 | 2.07x | faster | 139.526 | faster |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP + cuBLAS | 2 | 187.917 | 170.248 | 157.197 | 1.20x | faster | 148.918 | faster |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 571.616 | 480.821 | 233.894 | 2.44x | faster | 236.432 | faster |
| full step | width 128, layers 3, block 64, vocab 5256 | cuBLAS | 2 | 637.682 | 551.249 | 283.910 | 2.25x | faster | 278.922 | faster |

