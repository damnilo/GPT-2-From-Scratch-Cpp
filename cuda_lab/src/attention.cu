#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cudaAttentionForward(
    const float* K, const float* Q, const float* V, 
    float* K_t, float* out, float* scores,
    int batches, int sequence_len, int embedding_dim
) {
    float scale = 1/sqrtf(embedding_dim);
    dim3 block(1, 16, 16);
    dim3 grid(batches, (sequence_len + 15) / 16, (embedding_dim + 15) / 16);

    cudaBatchTranspose<<<grid, block>>>(K, K_t, batches, sequence_len, embedding_dim);
    
    dim3 dk_block(1, 16, 16);
    dim3 dk_grid(batches, (sequence_len + 15) / 16, (sequence_len + 15) / 16);
    
    cudaTiledBatchGemm<<<dk_grid, dk_block>>>(Q, K_t, scores, batches, sequence_len, sequence_len, embedding_dim);

    dim3 epilogue_block(1, 16, 1);
    dim3 epilogue_grid(batches, (sequence_len + 15) / 16, 1);
    
    cudaScaleMaskSoftmax<<<epilogue_grid, epilogue_block>>>(scores, batches, sequence_len, scale);

    dim3 av_block(1, 16, 16);
    dim3 av_grid(batches, (embedding_dim + 15) / 16, (sequence_len + 15) / 16);
    
    cudaTiledBatchGemm<<<av_grid, av_block>>>(scores, V, out, batches, sequence_len, embedding_dim, sequence_len);
}