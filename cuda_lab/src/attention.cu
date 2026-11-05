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
    
    cudaEpilogueForward<<<epilogue_grid, epilogue_block>>>(scores, batches, sequence_len, scale);

    dim3 av_block(1, 16, 16);
    dim3 av_grid(batches, (embedding_dim + 15) / 16, (sequence_len + 15) / 16);
    
    cudaTiledBatchGemm<<<av_grid, av_block>>>(scores, V, out, batches, sequence_len, embedding_dim, sequence_len);
}

void cudaAttentionBackward(
    const float* Q, const float* K, const float* V,
    float* V_t, float* grad_att_weights, float* grad_scores,
    float* grad_Q, float* grad_K, float* grad_V, float* grad_scores_t,
    float* scores, float* scores_t, float* grad_att, 
    int batches, int sequence_len, int embedding_dim
) {
    float scale = 1 / sqrtf(embedding_dim);
    dim3 sq_block(1, 16, 16);
    dim3 sq_grid(batches, (sequence_len + 15) / 16, (sequence_len + 15) / 16);

    cudaBatchTranspose<<<sq_grid, sq_block>>>(scores, scores_t, batches, sequence_len, sequence_len);

    dim3 bm_block(1, 16, 16);
    dim3 bm_grid(batches, (embedding_dim + 15) / 16, (sequence_len + 15) / 16);
    
    cudaTiledBatchGemm<<<bm_grid, bm_block>>>(scores_t, grad_att, grad_V, batches, sequence_len, embedding_dim, sequence_len);

    dim3 tr_block(1, 16, 16);
    dim3 tr_grid(batches, (sequence_len + 15) / 16, (embedding_dim + 15) / 16);

    cudaBatchTranspose<<<tr_grid, tr_block>>>(V, V_t, batches, sequence_len, embedding_dim);
    cudaTiledBatchGemm<<<sq_grid, sq_block>>>(grad_att, V_t, grad_att_weights, batches, sequence_len, sequence_len, embedding_dim);

    dim3 ep_block(1, 16, 1);
    dim3 ep_grid(batches, (sequence_len + 15) / 16, 1);

    cudaEpilogueBackward<<<ep_grid, ep_block>>>(grad_att_weights, scores, grad_scores, batches, sequence_len, scale);

    dim3 grad_block(1, 16, 16);
    dim3 grad_grid(batches, (embedding_dim + 15) / 16, (sequence_len + 15) / 16);

    cudaTiledBatchGemm<<<grad_grid, grad_block>>>(grad_scores, K, grad_Q, batches, sequence_len, embedding_dim, sequence_len);
    cudaBatchTranspose<<<sq_grid, sq_block>>>(grad_scores, grad_scores_t, batches, sequence_len, sequence_len);
    cudaTiledBatchGemm<<<grad_grid, grad_block>>>(grad_scores_t, Q, grad_K, batches, sequence_len, embedding_dim, sequence_len);
}