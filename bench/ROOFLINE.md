# Roofline

GPU: Tesla T4

Each cell is two warmups, then the median of 10 repeats. The compute roofs are the published T4 peaks: 8.1 TFLOP/s for FP32 and 65 TFLOP/s for FP16 tensor cores. The memory roof is 320 GB/s. FLOP/byte uses compulsory traffic: each input is read once and each output is written once. WMMA inputs are FP16. Fused attention reads K and V once for every query tile of 64. LayerNorm counts the mean and variance once per channel thread, which is the loop the kernel runs. Softmax counts five operations per score. A cell under 0.05 ms is marked launch: its percent of peak is not a throughput result. Above that line, memory means the intensity is below the ridge, and compute means it is above. These bytes are the algorithm's minimum traffic. Nsight Compute replaces them with measured DRAM bytes when the GPU allows the counters:

`ncu --section SpeedOfLight_Roofline ./build/roofline_bench`

| Kernel | Shape | ms | GFLOP/s | GB/s | FLOP/byte | % compute | % bandwidth | Limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cublas | 64x32x128 | 0.020 | 25.71 | 2.81 | 9.14 | 0.317 | 0.879 | launch |
| tiled | 64x32x128 | 0.014 | 37.97 | 4.15 | 9.14 | 0.469 | 1.298 | launch |
| wmma | 64x32x128 | 0.011 | 45.78 | 3.93 | 11.64 | 0.070 | 1.229 | launch |
| cublas | 64x128x512 | 0.024 | 351.30 | 17.84 | 19.69 | 4.337 | 5.575 | launch |
| tiled | 64x128x512 | 0.025 | 337.24 | 17.13 | 19.69 | 4.163 | 5.352 | launch |
| wmma | 64x128x512 | 0.015 | 550.24 | 18.27 | 30.12 | 0.847 | 5.709 | launch |
| cublas | 1024x1024x1024 | 0.515 | 4172.98 | 24.45 | 170.67 | 51.518 | 7.641 | compute |
| tiled | 1024x1024x1024 | 0.859 | 2499.67 | 14.65 | 170.67 | 30.860 | 4.577 | compute |
| wmma | 1024x1024x1024 | 0.994 | 2159.69 | 8.44 | 256.00 | 3.323 | 2.636 | compute |
| softmax | H=4 T=64 | 0.028 | 2.91 | 4.66 | 0.62 | 0.036 | 1.456 | launch |
| softmax | H=4 T=1024 | 2.156 | 9.73 | 15.57 | 0.62 | 0.120 | 4.864 | memory |
| layernorm | T=64 C=32 | 0.020 | 14.13 | 0.84 | 16.74 | 0.174 | 0.264 | launch |
| layernorm | T=64 C=128 | 0.053 | 79.72 | 1.25 | 64.00 | 0.984 | 0.389 | compute |
| layernorm | T=1024 C=128 | 0.391 | 174.35 | 2.68 | 64.94 | 2.153 | 0.839 | compute |
| scored attention | H=4 T=64 d=8 | 0.052 | 11.66 | 5.99 | 1.95 | 0.144 | 1.872 | memory |
| fused attention | H=4 T=64 d=8 | 0.065 | 9.27 | 0.50 | 18.50 | 0.114 | 0.157 | memory |
| scored attention | H=4 T=64 d=32 | 0.054 | 40.35 | 8.50 | 4.75 | 0.498 | 2.655 | memory |
| fused attention | H=4 T=64 d=32 | 0.134 | 16.28 | 0.98 | 16.62 | 0.201 | 0.306 | memory |
| scored attention | H=4 T=1024 d=8 | 2.538 | 61.15 | 26.75 | 2.29 | 0.755 | 8.360 | memory |
| fused attention | H=4 T=1024 d=8 | 0.881 | 176.10 | 5.06 | 34.82 | 2.174 | 1.580 | compute |
| scored attention | H=4 T=1024 d=32 | 2.539 | 219.70 | 27.67 | 7.94 | 2.712 | 8.647 | memory |
| fused attention | H=4 T=1024 d=32 | 2.004 | 278.37 | 8.90 | 31.29 | 3.437 | 2.780 | compute |
