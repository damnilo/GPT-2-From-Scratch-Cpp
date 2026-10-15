# CPU naspram CUDA

Procesor: AMD Ryzen 5 7530U with Radeon Graphics

Logički procesori: 12

Niti u OpenMP merenju: 12

GPU: nvcc nije nađen, CUDA redovi nisu popunjeni.

Prag za cuBLAS je proizvod dimenzija 65536 (`batch * rows * common * cols`). Manji proizvod ostaje na CPU-u, pa GPU kolona ostaje prazna. cuBLAS je označen kao brži samo kad mu je medijana manja od CPU medijane.

Svako merenje ima dva zagrevanja, pa medijanu od 10 ponavljanja. CPU kolona je OpenMP sa isključenim CUDA-om. Jedna nit je isti CPU kod sa `omp_set_num_threads(1)`. Ubrzanje je medijana jedne niti podeljena medijanom puta koji je stvarno izvršen.

Širina 32: pažnja `[1, 64, 8] @ [1, 8, 64]`, feed-forward `[64, 32] @ [32, 128]`, izlaz `[64, 32] @ [32, 5256]`.

Širina 128 i 4 glave: pažnja `[1, 64, 32] @ [1, 32, 64]`, feed-forward `[64, 128] @ [128, 512]`, izlaz `[64, 128] @ [128, 5256]`.

Model ima 3 bloka, kontekst 64 i rečnik 5256.

| Merenje | Dimenzije | Put | Niti | 1 nit ms | CPU ms | GPU ms | Ubrzanje prema 1 niti | cuBLAS naspram CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batchMatmul | [1, 64, 8] @ [1, 8, 64] | OpenMP | 12 | 0.486 | 0.317 | — | 1.53x | — |
| matmul | [64, 32] @ [32, 128] | OpenMP | 12 | 3.203 | 1.103 | — | 2.90x | — |
| matmul | [64, 32] @ [32, 5256] | OpenMP | 12 | 136.358 | 32.706 | — | 4.17x | — |
| batchMatmul | [1, 64, 32] @ [1, 32, 64] | OpenMP | 12 | 2.597 | 0.435 | — | 5.97x | — |
| matmul | [64, 128] @ [128, 512] | OpenMP | 12 | 52.034 | 15.304 | — | 3.40x | — |
| matmul | [64, 128] @ [128, 5256] | OpenMP | 12 | 422.304 | 65.257 | — | 6.47x | — |
| forward+backward | width 32, layers 3, block 64, vocab 5256 | OpenMP | 12 | 1199.790 | 511.912 | — | 2.34x | — |
| full step | width 32, layers 3, block 64, vocab 5256 | OpenMP | 12 | 1305.002 | 566.528 | — | 2.30x | — |
| forward+backward | width 128, layers 3, block 64, vocab 5256 | OpenMP | 12 | 3723.062 | 1352.886 | — | 2.75x | — |
| full step | width 128, layers 3, block 64, vocab 5256 | OpenMP | 12 | 4275.818 | 1498.724 | — | 2.85x | — |

