# Tensor-core GEMM

GPU: Tesla T4

A and B are FP32 and already on the device. Each cell is two warmups, then the median of 10 repeats. cuBLAS and the tiled kernel run in FP32. WMMA rounds A and B to FP16 before the timer, multiplies on tensor cores, and accumulates in FP32. The max abs diff is WMMA against cuBLAS on the same inputs. Training and generate stay on the FP32 path.

| Rows | Common | Cols | cuBLAS ms | Custom ms | WMMA ms | Max abs diff |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 16 | 16 | 0.017 | 0.013 | 0.010 | 2.033e-05 |
| 64 | 32 | 128 | 0.023 | 0.017 | 0.013 | 2.944e-05 |
| 64 | 128 | 512 | 0.027 | 0.035 | 0.019 | 5.917e-05 |
| 64 | 32 | 64 | 0.023 | 0.017 | 0.012 | 2.679e-05 |
