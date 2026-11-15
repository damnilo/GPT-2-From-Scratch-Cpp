# Fused attention

GPU: Tesla T4

Q, K, and V are already on the device. Each cell is two warmups, then the median of 10 repeats. The scored path is a transposed K, a product, a masked softmax, and a product with V. Its scratch, the transposed K and the score matrix, is allocated before the timer. The fused path does not allocate that scratch. Four heads. Training, generate, and the context of 64 stay on their current path.

| Head width | T | cuBLAS ms | Custom ms | Fused ms | Score bytes | Transposed K bytes | Extra bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 64 | 0.063 | 0.061 | 0.082 | 65536 | 8192 | 73728 |
| 8 | 128 | 0.116 | 0.116 | 0.149 | 262144 | 16384 | 278528 |
| 8 | 256 | 0.283 | 0.295 | 0.274 | 1048576 | 32768 | 1081344 |
| 8 | 512 | 0.851 | 0.916 | 0.517 | 4194304 | 65536 | 4259840 |
| 8 | 1024 | 3.259 | 3.393 | 1.166 | 16777216 | 131072 | 16908288 |
| 32 | 64 | 0.068 | 0.062 | 0.171 | 65536 | 32768 | 98304 |
| 32 | 128 | 0.120 | 0.117 | 0.313 | 262144 | 65536 | 327680 |
| 32 | 256 | 0.287 | 0.297 | 0.597 | 1048576 | 131072 | 1179648 |
| 32 | 512 | 0.873 | 0.923 | 0.627 | 4194304 | 262144 | 4456448 |
| 32 | 1024 | 1.783 | 1.921 | 1.551 | 16777216 | 524288 | 17301504 |
