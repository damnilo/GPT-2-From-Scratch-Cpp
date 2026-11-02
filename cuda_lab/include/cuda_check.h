#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)